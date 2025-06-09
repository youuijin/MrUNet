import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
import torch


class Mid_U_Net(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, out_layers=1):
        super(Mid_U_Net, self).__init__()
        chs = [32, 64, 64, 64]  # 기존보다 전체적으로 2배 확대
        self.out_layers = out_layers
        self.out_channels = out_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, chs[0], 3, padding=1),
            nn.InstanceNorm3d(chs[0]),
            nn.LeakyReLU(0.2),
            nn.Conv3d(chs[0], chs[0], 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(chs[0], chs[1], 3, stride=2, padding=1),
            nn.InstanceNorm3d(chs[1]),
            nn.LeakyReLU(0.2),
            nn.Conv3d(chs[1], chs[1], 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(chs[1], chs[2], 3, stride=2, padding=1),
            nn.InstanceNorm3d(chs[2]),
            nn.LeakyReLU(0.2),
            nn.Conv3d(chs[2], chs[2], 3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(chs[2], chs[3], 3, stride=2, padding=1),
            nn.InstanceNorm3d(chs[3]),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.dec1 = nn.Sequential(nn.Conv3d(chs[3], chs[3], 3, padding=1), nn.LeakyReLU(0.2))
        self.upsample1 = nn.ConvTranspose3d(chs[3], chs[2], 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(nn.Conv3d(chs[2]*2, chs[2], 3, padding=1), nn.LeakyReLU(0.2))
        self.upsample2 = nn.ConvTranspose3d(chs[2], chs[1], 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(nn.Conv3d(chs[1]*2, chs[1], 3, padding=1), nn.LeakyReLU(0.2))
        self.upsample3 = nn.ConvTranspose3d(chs[1], chs[0], 3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.Sequential(nn.Conv3d(chs[0]*2, 8, 3, padding=1), nn.LeakyReLU(0.2))

        # Output heads
        self.flows = nn.ModuleList([nn.Identity() for _ in range(4)])
        for i, res in enumerate([chs[3], chs[2], chs[1], 8]):
            if 4 - self.out_layers <= i:
                self.flows[i] = nn.Conv3d(res, out_channels, 3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        if self.out_channels == 6:
            means, stds = [], []

            x5 = self.dec1(x4)
            out = self.flows[0](x5)
            means.append(out[:, :3])
            stds.append(torch.exp(0.5 * out[:, 3:]))
            up_x5 = self.upsample1(x5)

            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            out = self.flows[1](x6)
            means.append(out[:, :3])
            stds.append(torch.exp(0.5 * out[:, 3:]))
            up_x6 = self.upsample2(x6)

            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            out = self.flows[2](x7)
            means.append(out[:, :3])
            stds.append(torch.exp(0.5 * out[:, 3:]))
            up_x7 = self.upsample3(x7)

            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            out = self.flows[3](x8)
            means.append(out[:, :3])
            stds.append(torch.exp(0.5 * out[:, 3:]))

            means = means[-self.out_layers:]
            stds = stds[-self.out_layers:]

            return self.combine_residuals(means), self.combine_residuals_std(stds), means, stds

        else:
            disp = []

            x5 = self.dec1(x4)
            disp.append(self.flows[0](x5))
            up_x5 = self.upsample1(x5)

            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            disp.append(self.flows[1](x6))
            up_x6 = self.upsample2(x6)

            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            disp.append(self.flows[2](x7))
            up_x7 = self.upsample3(x7)

            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            disp.append(self.flows[3](x8))

            disp = disp[-self.out_layers:]
            return self.combine_residuals(disp), disp

    def combine_residuals(self, flows):
        tot_flows = [flows[0]]
        for f in flows[1:]:
            prev = F.interpolate(tot_flows[-1], size=f.shape[2:], mode='trilinear', align_corners=True)
            tot_flows.append(prev + f)
        return tot_flows

    def combine_residuals_std(self, stds):
        tot_vars = [stds[0]]
        for s in stds[1:]:
            prev = F.interpolate(tot_vars[-1], size=s.shape[2:], mode='trilinear', align_corners=True)
            tot_vars.append(torch.sqrt(prev ** 2 + s ** 2))
        return tot_vars

if __name__ == '__main__':

    # 모델 정의 후 예시로 생성
    model = Mid_U_Net(in_channels=2, out_channels=3, out_layers=1)

    # 3D 입력: (channels=2, depth=64, height=64, width=64)
    summary(model, input_size=(2, 160, 192, 160), device="cpu")
