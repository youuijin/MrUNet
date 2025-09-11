import argparse
from pyprnt import prnt
from utils.trainer_utils import set_trainer
from utils.utils import print_with_timestamp

def main(args):
    trainer = set_trainer(args)
    print_with_timestamp(f"Start training: {trainer.log_name}")
    
    args_dict = vars(args)
    # prnt(args_dict)
    print_with_timestamp(f"Hyperparameters:")
    for key, value in args_dict.items():
        print_with_timestamp(f"    {key}: {value}")

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='OASIS', choices=['DLBS', 'OASIS', 'LUMIR', 'FDG_MRI', 'FDG_PET'])
    parser.add_argument("--model", type=str, default='U_Net', choices=['U_Net', 'Mid_U_Net', 'Big_U_Net'])
    parser.add_argument("--template_path", type=str, default="data/mni152_resample.npy")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--saved_path", default=None)

    # training options # TODO: 정리
    parser.add_argument("--pair_train", default=False, action='store_true')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--method", type=str, default='VM', choices=['VM', 'Mr', 'VM-Un', 'Mr-Un', 'VM-diff', 'Mr-diff', 'VM-Un-diff', 'VM-Al-Un', 'VM-SFA', 'VM-SFA-diff', 'VM-SFAeach', 'VM-SFAeach-diff', 'Mr-SFAeach', 'Mr-SFAeach-diff'])
    parser.add_argument("--loss", type=str, default="MSE", choices=['NCC', 'MSE']) #TODO: add Uncertainty version
    parser.add_argument("--numpy", action='store_true', default=False)
    
    # for regularizer
    parser.add_argument("--reg", type=str, default=None)
    parser.add_argument("--alpha", type=str, default=None)
    parser.add_argument("--p", type=float, default=0.0)
    parser.add_argument("--alp_sca", type=float, default=1.0)
    parser.add_argument("--sca_fn", type=str, default='exp', choices=['exp', 'linear'])
    parser.add_argument("--only_kl", action='store_true', default=False)

    # for uncertainty
    parser.add_argument("--image_sigma", type=float, default=0.02)
    parser.add_argument("--prior_lambda", type=float, default=20.0)
    parser.add_argument("--num_samples", type=int, default=5)

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_num", type=int, default=2)
    # parser.add_argument("--val_detail", default=False, action='store_true')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'multistep'])
    parser.add_argument("--lr_milestones", type=str, default=None)

    # log options
    parser.add_argument("--wandb_name", type=str, default='rebrain')
    parser.add_argument("--log_dir", type=str, default='logs')

    args = parser.parse_args()

    main(args)
