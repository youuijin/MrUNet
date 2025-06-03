from utils.dataset import set_dataloader
from networks.U_Net import U_Net
from networks.VecInt import VecInt
import csv, torch

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
        self.model = U_Net(out_channels=self.out_channels, out_layers=self.out_layers)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.cuda()
        self.model.eval()
        # set data
        _, _, self.save_loader = set_dataloader(args.image_path, args.template_path, 1)

    def save_results(self, csv_path, row):
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(row)

    def check_tested(self, model_path):
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            existing_values = set(row[0] for row in reader)

        # 3. 값이 없으면 추가
        if model_path.split('/')[-1] not in existing_values:
            return False
        else:
            return True

