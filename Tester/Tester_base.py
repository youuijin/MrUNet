from utils.dataset import set_dataloader, set_dataloader_usingcsv, set_paired_dataloader_usingcsv
from networks.U_Net import U_Net
from networks.Big_U_Net import Mid_U_Net
from networks.VecInt import VecInt
import csv, torch, os

import nibabel as nib

class Tester:
    def __init__(self, model_path, args):
        # check already tested
        tested = self.check_tested(model_path)
        self.already_tested = False
        if tested:
            self.already_tested = True
            return

        # set method
        self.method = model_path.split("/")[-1].split("_")[0]
        self.log_name = model_path.split("/")[-1]
        if self.method == 'VM' or self.method == 'VM-diff' :
            self.out_channels = 3
            self.out_layers = 1
            if 'diff' in self.method:
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        elif self.method in ['VM-Un', 'VM-Un-diff', 'VM-Al-Un', 'VM-Al-Un-diff', 'VM-SFA', 'VM-SFAeach', 'VM-SFAeach-diff']:
            self.out_channels = 6
            self.out_layers = 1
            if 'diff' in self.method:
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)
        
        elif self.method in ['Mr', 'Mr-diff', 'Mr-res', 'Mr-diff-res', 'Mr-diff-ressca']:
            self.out_channels = 3
            self.out_layers = 3
            if 'diff' in self.method:
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        elif self.method in ['Mr-Un', 'Mr-Un-res', 'Mr-Un-reskl', 'Mr-Un-diff', 'Mr-Un-diff-res', 'Mr-Un-diff-ressca']:
            self.out_channels = 6
            self.out_layers = 3
            if 'diff' in self.method:
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        # set model
        self.model_type = model_path.split("/")[-3]
        if self.model_type == 'Mid_U_Net':
            self.model = Mid_U_Net(out_channels=self.out_channels, out_layers=self.out_layers)
        else:
            self.model = U_Net(out_channels=self.out_channels, out_layers=self.out_layers)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.cuda()
        self.model.eval()
        # set data
        if args.pair_test:
            _, _, self.save_loader = set_paired_dataloader_usingcsv(self.test_dataset, 'data/data_list', batch_size=1, numpy=False)
        else:
            _, _, self.save_loader = set_dataloader_usingcsv(self.test_dataset, 'data/data_list', args.template_path, 1, numpy=False)

    def set_dataset(self, args):
        assert args.test_dataset in [None, 'OASIS', 'DLBS', 'FDG_MRI', 'FDG_PET']
        # set train dataset
        if (args.model_dir and 'DLBS' in args.model_dir) or (args.model_path and 'DLBS' in args.model_path):
            self.train_dataset = 'DLBS'
        else:
            self.train_dataset = 'OASIS'

        # set test dataset
        if args.test_dataset is None:
            # add dataset 
            self.test_dataset = self.train_dataset
        else:
            self.test_dataset = args.test_dataset
        
        if self.test_dataset == 'OASIS':
            self.label_path = 'data/OASIS_label_core'
        elif self.test_dataset == 'DLBS':
            self.label_path = 'data/DLBS_label_core'
        elif self.test_dataset in ['FDG_MRI', 'FDG_PET']:
            self.label_path = 'data/FDG_label'

        self.csv_dir = f'{args.csv_dir}/train_{self.train_dataset}/test_{self.test_dataset}'
                
    
    def save_results(self, csv_path, row):
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(row)

    def check_tested(self, model_path):
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # 헤더 건너뛰기
                existing_values = set()
                for row in reader:
                    existing_values.add(row[1])
                    existing_values.add(row[2])

            # 3. 값이 없으면 추가
            if model_path.split('/')[-1] not in existing_values:
                return False
            else:
                print('Already Tested:', model_path)
                return True
        except:
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['train_dataset','model','log_name','ICC','slope','y-intercept','R2'])
            return False

    def load_single_template(self, path):
        img = nib.load(path)
        img = img.get_fdata()

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#

        return torch.tensor(img, dtype=torch.float32)
        
