from pytorch_lightning import Trainer
import torch
from pytorchtools.MINCDataModule import MINCDataModule
from pytorchtools.arg_parser import ArgParser
from pytorchtools.patch_classifier import PatchClassifier
import warnings


def main():
    warnings.filterwarnings('ignore')
    # Start training from scratch
    if not args.resume and not args.test:
        # Parse the argements
        json_data = arg_parser.json_data
        model = PatchClassifier(json_data)
        dm = MINCDataModule(args.data_root, json_data)
        if torch.cuda.device_count() >= 1:
            trainer = Trainer(progress_bar_refresh_rate=20, log_every_n_steps=20, flush_logs_every_n_steps=800,
                              max_epochs=args.epochs, replace_sampler_ddp=False)
        else:
            trainer = Trainer(progress_bar_refresh_rate=1, log_every_n_steps=1, flush_logs_every_n_steps=1,
                              max_epochs=3, replace_sampler_ddp=False)
        trainer.fit(model, dm)

        # run test set
        trainer.test()


if __name__ == '__main__':
    arg_parser = ArgParser()
    args = arg_parser.args
    if not args.net_list:
        main()
