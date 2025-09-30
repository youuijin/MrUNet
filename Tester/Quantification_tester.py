from Tester.Tester_base import Tester
from utils.dataset import set_dataloader_usingcsv, set_paired_dataloader_usingcsv
from utils.utils import apply_deformation_using_disp
from sklearn.metrics import r2_score
import torch, os, csv
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Quantification_tester(Tester):
    # need PET, paired MRI segmentation
    def __init__(self, model_path, args):
        self.set_dataset(args)
        self.test_dataset = 'FDG_MRI'
        self.pet_dir = 'data/FDG_PET_percent'
        
        self.label_path = args.label_path
        if args.label_path == "data/FDG_label_cortex":
            self.dice_type = 'dice6'
            self.seg_num = 6
        else:
            self.dice_type = 'dice35'
            self.seg_num = 35
        self.csv_dir = f'{args.csv_dir}/{self.train_dataset}/quant/quant_{self.seg_num}'
        self.csv_path = f"{self.csv_dir}/quant_global.csv"
        self.plot = True

        super().__init__(model_path, args)
        
        self.trained_dataset = model_path.split("/")[-4]

        if args.pair_test:
            _, _, self.save_loader = set_paired_dataloader_usingcsv(self.test_dataset, 'data/data_list', batch_size=1, numpy=False, return_path=True)
        else:
            _, _, self.save_loader = set_dataloader_usingcsv(self.test_dataset, 'data/data_list', 'data/mni152_resample.nii', batch_size=1, numpy=False, return_path=True)
        
        if self.dice_type == 'dice6':
            self.label_name = {
                1: 'frontal',
                2: 'parietal',
                3: 'temporal',
                4: 'occipital',
                5: 'subcortical',
                6: 'cerebellum'
            }
        else:
            # cerebellum cortex : 6, 24
            # remove : 18, 34
            self.label_name = {
                1: 'Left-Cerebral-White-Matter',
                2: 'Left-Cerebral-Cortex',
                3: 'Left-Lateral-Ventricle',
                4: 'Left-Inf-Lat-Vent',
                5: 'Left-Cerebellum-Exterior',
                6: 'Left-Cerebellum-Cortex',
                7: 'Left-Thalamus',
                8: 'Left-Caudate',
                9: 'Left-Putamen',
                10: 'Left-Pallidum',
                11: '3rd-Ventricle',
                12: '4th-Ventricle',
                13: 'Brain-Stem',
                14: 'Left-Hippocampus',
                15: 'Left-Amygdala',
                16: 'Left-Accumbens-area',
                17: 'Left-VentralDC',
                19: 'Left-choroid-plexus',
                20: 'Right-Cerebral-White-Matter',
                21: 'Right-Cerebral-Cortex',
                22: 'Right-Lateral-Ventricle',
                23: 'Right-Inf-Lat-Vent',
                24: 'Right-Cerebellum-White-Matter',
                25: 'Right-Cerebellum-Cortex',
                26: 'Right-Thalamus',
                27: 'Right-Caudate',
                28: 'Right-Putamen',
                29: 'Right-Pallidum',
                30: 'Right-Hippocampus',
                31: 'Right-Amygdala',
                32: 'Right-Accumbens-area',
                33: 'Right-VentralDC',
                35: 'Right-choroid-plexus'
            }

        self.plot_save_dir = f"visualization/R2_plot/{'/'.join(model_path.split('/')[1:])}"
        if self.dice_type == 'dice35':
            self.plot_save_dir = f"visualization/R2_plot_35/{'/'.join(model_path.split('/')[1:])}"
        os.makedirs(self.plot_save_dir, exist_ok=True)

    def calc_suv(self, seg, img, labels):
        if type(labels) == int:
            labels = [labels]
        seg = seg.int()
        mask = torch.zeros_like(seg, dtype=torch.bool)
        for l in labels:
            mask |= (seg == l)
        intersection = (mask * img).sum().float()
        return intersection / mask.sum()

    # def r2_corr(self, x, y):
    #     x = np.asarray(x, dtype=float)
    #     y = np.asarray(y, dtype=float)
    #     # 유효값만 필터
    #     valid = np.isfinite(x) & np.isfinite(y)
    #     x, y = x[valid], y[valid]
    #     if x.size < 2 or y.size < 2:
    #         return np.nan
    #     # r = np.corrcoef(x, y)[0, 1]   # 피어슨 상관계수
    #     r2 = r2_score(y, x)
    #     return r2
    def regression_params(self, x, y):
        x = np.asarray(x).reshape(-1,1)
        y = np.asarray(y)
        m = LinearRegression().fit(x, y)
        return m.coef_[0], m.intercept_, m.score(x, y)  # slope, intercept, R²
    
    def icc_matrix(self, X, icc_type="ICC2_1"):
        """
        X: shape [N, k]  (N=피험자 수, k=방법/측정자 수)
        icc_type: "ICC2_1" (absolute agreement), "ICC3_1" (consistency)
        반환: float (ICC 값)
        참고: McGraw & Wong (1996), Shrout & Fleiss 표기
        """
        X = np.asarray(X, dtype=float)
        # NaN이 있는 행은 제거 (complete-case)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        n, k = X.shape
        if n < 2 or k < 2:
            raise ValueError("피험자>=2, 방법>=2 필요합니다.")

        # 평균들
        mean_subject = X.mean(axis=1, keepdims=True)   # [n,1]
        mean_rater   = X.mean(axis=0, keepdims=True)   # [1,k]
        grand_mean   = X.mean()

        # 제곱합(2원 분산분석)
        ss_total = ((X - grand_mean) ** 2).sum()
        ss_rows  = (k * ((mean_subject - grand_mean) ** 2)).sum()  # subjects 효과
        ss_cols  = (n * ((mean_rater   - grand_mean) ** 2)).sum()  # raters/방법 효과
        ss_err   = ss_total - ss_rows - ss_cols

        # 자유도
        df_rows = n - 1
        df_cols = k - 1
        df_err  = (n - 1) * (k - 1)

        # 평균제곱
        ms_rows = ss_rows / df_rows     # MSR (subjects)
        ms_cols = ss_cols / df_cols     # MSC (raters)
        ms_err  = ss_err  / df_err      # MSE (residual)

        t = icc_type.upper()
        if t in ["ICC2_1", "ICC(2,1)", "A,1", "ICC2"]:
            # Two-way random, absolute agreement, single rater
            icc = (ms_rows - ms_err) / (ms_rows + (k - 1)*ms_err + (k*(ms_cols - ms_err)/n))
        elif t in ["ICC3_1", "ICC(3,1)", "C,1", "ICC3"]:
            # Two-way mixed, consistency, single rater
            icc = (ms_rows - ms_err) / (ms_rows + (k - 1)*ms_err)
        else:
            raise ValueError(f"지원하지 않는 icc_type: {icc_type}")
        return float(icc)

    def icc_two_vectors(self, a, b, icc_type="ICC2_1"):
        """
        편의함수: 두 방법(예: individual vs template)만 있을 때
        a, b: 길이 N의 1D 배열
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        X = np.stack([a, b], axis=1)  # [N,2]
        return self.icc_matrix(X, icc_type=icc_type)


    def test(self):
        if self.dice_type == 'dice6':
            temp_seg = nib.load('data/FDG_label_cortex/template_T1w_MRI.nii.gz').get_fdata()
        else:
            temp_seg = nib.load('data/mni152_label.nii').get_fdata()
        temp_seg = torch.tensor(temp_seg).unsqueeze(0).unsqueeze(0).cuda()

        origin_SUVrs = [[] for _ in range(self.seg_num)] # for all labels + global
        moved_SUVrs = [[] for _ in range(self.seg_num)] # for all labels + global
        cnt = 0

        for img, template, _, _, _, path in tqdm(self.save_loader):
            # img, template: MR images
            path = path[0]
            sub_name = path.split('/')[-1].split("_")[0] # sub-N
            seg_path = f"{self.label_path}/{sub_name}_T1w_MRI.nii.gz"
            pet_path = f"{self.pet_dir}/core_{sub_name}_FDG_PET.nii.gz"

            if not os.path.exists(seg_path) or not os.path.exists(pet_path):
                print(seg_path, pet_path)
                print("No Segments or PET scans")
                continue
            cnt += 1

            seg = nib.load(seg_path).get_fdata().astype(np.float32)
            seg = torch.tensor(seg).unsqueeze(0).unsqueeze(0).cuda()
            pet = nib.load(pet_path).get_fdata().astype(np.float32)
            pet = torch.tensor(pet).unsqueeze(0).unsqueeze(0).cuda()

            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            disp = self.model(stacked_input)[0][-1]
            if 'diff' in self.method:
                disp = self.integrate(disp)

            deformed_pet = apply_deformation_using_disp(pet, disp) # deformed PET
            # deformed_seg = apply_deformation_using_disp(seg, disp, mode='nearest')


            # calc reference region (6)
            if self.seg_num == 6:
                ref_num = [6]
            else:
                ref_num = [6, 24]
            origin_suv_ref = self.calc_suv(seg, pet, ref_num)
            moved_suv_ref = self.calc_suv(temp_seg, deformed_pet, ref_num)

            for label in tqdm(self.label_name, position=1, leave=False): # label: 1, 2, 3, 4, 5
                origin_suv = self.calc_suv(seg, pet, label)
                moved_suv = self.calc_suv(temp_seg, deformed_pet, label)
                origin_SUVrs[label-1].append((origin_suv/origin_suv_ref).item())
                moved_SUVrs[label-1].append((moved_suv/moved_suv_ref).item())

            # calculate global
            origin_suv = self.calc_suv(seg, pet, [i for i in self.label_name if i not in ref_num])
            moved_suv = self.calc_suv(temp_seg, deformed_pet, [i for i in self.label_name if i not in ref_num])
            origin_SUVrs[self.seg_num-1].append((origin_suv/origin_suv_ref).item())
            moved_SUVrs[self.seg_num-1].append((moved_suv/moved_suv_ref).item())

        iccs = []

        if self.seg_num == 6:
            # calculate regression parameter
            for idx, label_name in enumerate(['frontal', 'parietal', 'temporal', 'occipital', 'subcortical', 'global']):
                o, m = origin_SUVrs[idx], moved_SUVrs[idx]
                slope, y_inter, r2_value = self.regression_params(o, m)
                icc = self.icc_two_vectors(o, m)
                iccs.append(icc)

                results = [self.trained_dataset, self.model_type, self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
                if not os.path.exists(f'{self.csv_dir}/quant_{label_name}.csv'):
                    header = ['train_dataset','model','log_name','ICC','slope','y-intercept','R2']
                    with open(f'{self.csv_dir}/quant_{label_name}.csv', mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                self.save_results(f'{self.csv_dir}/quant_{label_name}.csv', results)

            iccs = np.array(iccs)

            for label, (o, m, icc) in enumerate(zip(origin_SUVrs, moved_SUVrs, iccs)):
                if label == 5:
                    region_name = 'global'
                else:
                    region_name = self.label_name[label+1]
                self.save_plot(o, m, region_name, icc)
            
        else:
            iccs = [0. for _ in range(self.seg_num)]
            # calculate regression parameter
            for idx, label_name in self.label_name.items():
                if idx in ref_num:
                    continue
                o, m = origin_SUVrs[idx-1], moved_SUVrs[idx-1]
                slope, y_inter, r2_value = self.regression_params(o, m)
                icc = self.icc_two_vectors(o, m)
                iccs[idx-1] = icc

                results = [self.trained_dataset, self.model_type, self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
                if not os.path.exists(f'{self.csv_dir}/quant_{label_name}.csv'):
                    header = ['train_dataset','model','log_name','ICC','slope','y-intercept','R2']
                    with open(f'{self.csv_dir}/quant_{label_name}.csv', mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                self.save_results(f'{self.csv_dir}/quant_{label_name}.csv', results)
           
            # for global  
            o, m = origin_SUVrs[self.seg_num-1], moved_SUVrs[self.seg_num-1]
            slope, y_inter, r2_value = self.regression_params(o, m)
            icc = self.icc_two_vectors(o, m)
            iccs[self.seg_num-1] = icc

            results = [self.trained_dataset, self.model_type, self.log_name, round(icc, 5), round(slope, 5), round(y_inter, 5), round(r2_value, 5)]
            if not os.path.exists(f'{self.csv_dir}/quant_global.csv'):
                header = ['train_dataset','model','log_name','ICC','slope','y-intercept','R2']
                with open(f'{self.csv_dir}/quant_global.csv', mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
            self.save_results(f'{self.csv_dir}/quant_global.csv', results)

            iccs = np.array(iccs)

            for idx, label_name in self.label_name.items():
                if idx in ref_num:
                    continue
                self.save_plot(origin_SUVrs[idx-1], moved_SUVrs[idx-1], label_name, iccs[idx-1])

            self.save_plot(origin_SUVrs[self.seg_num-1], moved_SUVrs[self.seg_num-1], 'global', iccs[self.seg_num-1])
        
    def save_plot(self, origin, moved, region_name, icc):
        origin = np.asarray(origin, dtype=float)
        moved = np.asarray(moved, dtype=float)
        plt.figure(figsize=(6,6))
        # (1) Scatter plot
        plt.scatter(origin, moved, color='blue', s=50, alpha=0.7, label="SUVr values")

        # (2) Linear regression line
        coef = np.polyfit(origin, moved, 1)
        poly_fn = np.poly1d(coef)
        x_line = np.linspace(origin.min(), origin.max(), 100)
        plt.plot(x_line, poly_fn(x_line), color='red', linewidth=2, label=f"Fit line (ICC={icc:.2f})")

        # (3) y=x 점선
        plt.plot(x_line, x_line, 'k--', linewidth=1.5, label="y=x")

        # 축 레이블 및 범례
        plt.xlabel("SUVr in individual space")
        plt.ylabel("SUVr in template space")
        plt.title(f"Correlation of SUVr {region_name}")
        plt.legend()
        plt.axis("equal")   # x, y 축 스케일 같게
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(f'{self.plot_save_dir}/{region_name}.png')
        plt.close()

    # def save_plot_using_cmap(self, origin, moved, dices, label_num, r2):
    #     origin = np.asarray(origin, dtype=float)
    #     moved  = np.asarray(moved, dtype=float)
    #     dices  = np.asarray(dices, dtype=float)  # 0~1 Dice score

    #     plt.figure(figsize=(6,6))

    #     # (1) Scatter plot with Dice as color
    #     sc = plt.scatter(
    #         origin, moved,
    #         c=dices, cmap="viridis",  # 추천 colormap
    #         vmin=0, vmax=1,           # Dice는 0~1 범위
    #         s=50, alpha=0.8,
    #         edgecolors="k", linewidth=0.3
    #     )

    #     # (2) Linear regression line
    #     coef = np.polyfit(origin, moved, 1)
    #     poly_fn = np.poly1d(coef)
    #     x_line = np.linspace(origin.min(), origin.max(), 100)
    #     plt.plot(x_line, poly_fn(x_line), color='red', linewidth=2, label=f"Fit line (R²={r2:.2f})")

    #     # (3) y=x 점선
    #     plt.plot(x_line, x_line, 'k--', linewidth=1.5, label="y=x")

    #     region_name = self.label_name[label_num]

    #     # 축/제목
    #     plt.xlabel("SUVr in individual space")
    #     plt.ylabel("SUVr in template space")
    #     plt.title(f"Correlation of SUVr {region_name}")

    #     # Color bar for Dice
    #     cbar = plt.colorbar(sc)
    #     cbar.set_label("Dice score")

    #     plt.legend()
    #     plt.axis("equal")
    #     plt.grid(True, linestyle="--", alpha=0.6)

    #     plt.tight_layout()
    #     plt.savefig(f"{self.plot_save_dir}/{region_name}_cmp.png", dpi=300)
    #     plt.close()
