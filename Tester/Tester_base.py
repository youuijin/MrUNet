from utils.dataset import set_dataloader, set_dataloader_usingcsv
from networks.U_Net import U_Net
from networks.Big_U_Net import Mid_U_Net
from networks.VecInt import VecInt
import csv, torch

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
            if self.method == 'VM-diff':
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        elif self.method == 'VM-Un' or self.method == 'VM-Un-diff':
            self.out_channels = 6
            self.out_layers = 1
            if self.method == 'VM-Un-diff':
                self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)
        
        elif self.method == 'Mr' or self.method == 'Mr-diff':
            self.out_channels = 3
            self.out_layers = 3
            if self.method == 'Mr-diff':
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
        # _, _, self.save_loader = set_dataloader(args.image_path, args.template_path, 1)
        _, _, self.save_loader = set_dataloader_usingcsv(self.test_dataset, 'data/data_list', args.template_path, 1)

    def set_dataset(self, args):
        # add dataset 
        if (args.model_dir and 'DLBS' in args.model_dir) or (args.model_path and 'DLBS' in args.model_path):
            self.train_dataset = 'DLBS'
            if args.external:
                self.test_dataset = 'OASIS'
            else:
                self.test_dataset = "DLBS"
        else:
            self.train_dataset = 'OASIS'
            if args.external:
                self.test_dataset = "DLBS"
            else:
                self.test_dataset = 'OASIS'

        if self.test_dataset == 'OASIS':
            self.label_path = 'data/OASIS_label_core'
        elif self.test_dataset == 'DLBS':
            self.label_path = 'data/DLBS_label_35'
    
    def save_results(self, csv_path, row):
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(row)

    def check_tested(self, model_path):
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            existing_values = set(row[1] for row in reader)

        # 3. 값이 없으면 추가
        if model_path.split('/')[-1] not in existing_values:
            return False
        else:
            print('Already Tested:', model_path)
            return True

    def load_single_template(self, path):
        img = nib.load(path)
        img = img.get_fdata()

        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0,1]#

        return torch.tensor(img, dtype=torch.float32)
        
