import torch, random
import torch.nn.functional as F
import numpy as np

def clamp01(x):
    if isinstance(x, torch.Tensor):
        return x.clamp(0.0, 1.0)
    elif isinstance(x, np.ndarray):
        return np.clip(x, 0.0, 1.0)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

# -------- 개별 증강(범위는 나중에 strength로 스케일) --------
def aug_brightness_contrast_gamma(x, a_rng, b_rng, g_rng, center='mean'):
    a = random.uniform(*a_rng)  # contrast
    b = random.uniform(*b_rng)  # brightness
    g = random.uniform(*g_rng)  # gamma
    m = x.mean() if center=='mean' else (x.median() if center=='median' else 0.5)
    y = a*(x - m) + m + b
    y = clamp01(y)
    y = torch.pow(y, g)
    return clamp01(y)

def aug_bias_field(x, alpha=0.12):
    """
    2차 다항식 a + bX + cY + dZ + eXY + fXZ + gYZ + hX^2 + ... 로 만든 매우 부드러운 bias.
    alpha: 진폭(0~0.2 권장)
    """
    squeeze = False
    if x.dim()==2:
        H,W = x.shape; D=1
        X,Y = torch.meshgrid(torch.linspace(-1,1,H, device=x.device), 
                             torch.linspace(-1,1,W, device=x.device), indexing='ij')
        terms = [torch.ones_like(X), X, Y, X*Y, X*X, Y*Y]
        rand_coefs = [torch.randn(1, device=x.device, dtype=x.dtype) for _ in terms]
        field = sum(c*t for c,t in zip(rand_coefs, terms))
        field = field[None,None]           # [1,1,H,W]
        x_in = x[None,None]; squeeze=True
    elif x.dim()==3:
        D,H,W = x.shape
        Z,X,Y = torch.meshgrid(
            torch.linspace(-1,1,D, device=x.device),
            torch.linspace(-1,1,H, device=x.device),
            torch.linspace(-1,1,W, device=x.device),
            indexing='ij'
        )
        terms = [torch.ones_like(Z), X, Y, Z, X*Y, X*Z, Y*Z, X*X, Y*Y, Z*Z]
        rand_coefs = [torch.randn(1, device=x.device, dtype=x.dtype) for _ in terms]
        field = sum(c*t for c,t in zip(rand_coefs, terms))  # [D,H,W]
        field = field[None,None]         # [1,1,D,H,W]
        x_in = x[None,None]; squeeze=True
    else:
        raise ValueError("Use 2D or 3D tensor")

    # 0-mean, ±alpha로 스케일 → 1±alpha 범위
    field = field - field.mean()
    field = field / (field.abs().amax() + 1e-6)
    field = 1.0 + alpha * field

    y = x_in * field
    y = torch.clamp(y, 0, 1)
    if squeeze: y = y.squeeze(0).squeeze(0)
    return y


def aug_piecewise_hist_warp(x, k, jitter):
    xp = torch.linspace(0,1,k+2, device=x.device, dtype=x.dtype)
    fp = xp.clone()
    fp[1:-1] = torch.clamp(fp[1:-1] + (torch.rand(k, device=x.device, dtype=x.dtype)-0.5)*2*jitter, 0,1)
    fp, _ = torch.sort(fp)
    flat = x.flatten()
    idx = torch.bucketize(flat, xp) - 1; idx = idx.clamp(0, xp.numel()-2)
    x0, x1 = xp[idx], xp[idx+1]; y0, y1 = fp[idx], fp[idx+1]
    t = (flat - x0)/(x1 - x0 + 1e-6)
    out = torch.lerp(y0, y1, t)
    return clamp01(out.view_as(x))

def renorm_percentile(x, p_low=1.0, p_high=99.0):
    v, _ = torch.sort(x.flatten())
    def p(q):
        k = (q/100.0)*(v.numel()-1)
        i0 = int(torch.floor(k).clamp(0, v.numel()-1)); i1 = min(i0+1, v.numel()-1)
        w = float(k - i0); return (1-w)*v[i0] + w*v[i1]
    lo, hi = p(p_low), p(p_high)
    return clamp01((torch.clamp(x, lo, hi) - lo) / (hi - lo + 1e-6))

# -------- 통합 증강 파이프라인 (strength로 조절) --------
class IntensityAug:
    """
    strength ∈ [0,1]:
      0   → 거의 없음
      0.3 → light
      0.5 → medium
      0.8 → heavy
    마지막에 percentile로 재정규화해서 항상 [0,1] 유지.
    """
    def __init__(self, strength=0.5, p_low=1.0, p_high=99.0):
        self.s = float(max(0.0, min(1.0, strength)))
        self.p_low, self.p_high = p_low, p_high

        # 확률도 함께 스케일
        self.p_bc_gamma = 0.30 + 0.70*self.s      # 0.3~1.0
        self.p_bias     = 0.20 + 0.60*self.s      # 0.2~0.8
        self.p_histwarp = 0.10 + 0.40*self.s      # 0.1~0.5

        # 범위 스케일: “최대 폭 × s”
        self.a_rng = (1.0 - 0.40*self.s, 1.0 + 0.40*self.s)      # contrast
        self.b_rng = (-0.20*self.s, 0.20*self.s)                 # brightness
        self.g_rng = (1.0 - 0.50*self.s, 1.0 + 0.50*self.s)      # gamma
        self.bias_alpha = 0.04 + (0.12 - 0.04) * (self.s ** 1.2)
        self.hist_k = max(1, int(1 + 3*self.s))                   # control points
        self.hist_jitter = 0.05 + 0.25*self.s

    def clamp01(x):
        if isinstance(x, torch.Tensor):
            return x.clamp(0.0, 1.0)
        elif isinstance(x, np.ndarray):
            return np.clip(x, 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def __call__(self, x):
        y = x
        if random.random() < self.p_bc_gamma:
            y = aug_brightness_contrast_gamma(y, self.a_rng, self.b_rng, self.g_rng)
        if random.random() < self.p_bias:
            y = aug_bias_field(y, self.bias_alpha)
        if random.random() < self.p_histwarp:
            y = aug_piecewise_hist_warp(y, self.hist_k, self.hist_jitter)
        # 항상 0-1 재정규화
        return clamp01(y)
