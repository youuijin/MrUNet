# from Trainer.Basic_trainer import Basic_Trainer
from Trainer.VoxelMorph_trainer import VoxelMorph_Trainer
from Trainer.VoxelMorph_Uncert_trainer import VoxelMorph_Uncert_Trainer
from Trainer.MrRegNet_trainer import MrRegNet_Trainer
# from Trainer.SyN_trainer import SyN_Trainer

def set_trainer(args):
    if args.method == "VM" or args.method == "VM-diff":
        trainer = VoxelMorph_Trainer(args)
    elif args.method == "VM-Un" or args.method == 'VM-Un-diff':
        trainer = VoxelMorph_Uncert_Trainer(args)
    elif args.method == "Mr" or args.method =='Mr-diff':
        trainer = MrRegNet_Trainer(args)
    # elif args.method == "Mr-Un":
    #     trainer = MrRegNet_Trainer(out_layers=args.out_layers, out_channels=6, loss=args.loss, reg=args.reg, img_sigma=args.img_sigma, prior_lambda=args.prior_lambda) 
    #     #TODO: Scaling accross multi-resolution level
    # elif args.method == "SyN":
    #     trainer = SyN_Trainer()
    else:
        raise ValueError("Error! Undefined Method:", args.method)
    
    return trainer