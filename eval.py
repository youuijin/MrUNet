import argparse
from pyprnt import prnt
import torch, os

from Tester.DSC_tester import DSC_Tester
from Tester.Folding_tester import Folding_Tester
from Tester.Similarity_tester import Similarity_Tester
from Tester.Blur_tester import Blur_Tester
from Tester.Affine_tester import Affine_tester
from Tester.SyN_tester import SyN_tester
from Tester.Quantification_tester import Quantification_tester

from utils.utils import set_seed

def set_tester(test_method, model_path, args):
    if test_method == 'dice':
        tester = DSC_Tester(model_path, args)
    elif test_method == 'folding':
        tester = Folding_Tester(model_path, args)
    elif test_method == 'similar':
        tester = Similarity_Tester(model_path, args)
    elif test_method == 'blur':
        tester = Blur_Tester(model_path, args)
    elif test_method == 'quant':
        tester = Quantification_tester(model_path, args)

    return tester

def main(args):
    if args.tester == 'SyN':
        tester = SyN_tester('SyN', args)
        with torch.no_grad():
            tester.test()
            return
    elif args.tester == 'Affine':
        tester = Affine_tester('Affine_only', args)
        with torch.no_grad():
            tester.test()
            return
    else:
        if args.model_path is not None:
            paths = [args.model_path]
        else:
            if os.path.isdir(args.model_dir):
                paths = os.listdir(args.model_dir)
                paths = [f'{args.model_dir}/{p}' for p in paths]
            else:
                print("No directory:", args.model_dir)
                return

    for model_path in paths:
        set_seed()
        tester = set_tester(args.test_method, model_path, args)
        if tester.already_tested:
            continue
        print("Start evaluation:", tester.log_name)
        print("Evaluation method:", args.test_method)

        args_dict = vars(args)
        prnt(args_dict)
        with torch.no_grad():
            tester.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tester", choices=[None, 'Affine', 'SyN'], default=None)
    parser.add_argument("--test_method", choices=['dice', 'folding', 'similar', 'blur', 'quant'], default='similar')
    parser.add_argument("--pair_test", action='store_true', default=False)

    parser.add_argument("--template_path", type=str, default="data/mni152_resample.nii")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default='results/saved_models/OASIS/completed')
    parser.add_argument("--csv_dir", type=str, default="results/template/csvs")

    parser.add_argument("--external", action='store_true', default=False)

    # for Affine & SyN Tester
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    
    parser.add_argument("--save_num", type=int, default=0)

    args = parser.parse_args()

    main(args)
