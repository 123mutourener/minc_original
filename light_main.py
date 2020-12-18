from pytorch_lightning import Trainer

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
        trainer = Trainer(progress_bar_refresh_rate=1, log_every_n_steps=1, flush_logs_every_n_steps=1,
                          max_epochs=args.epochs)
        trainer.fit(model, dm)


if __name__ == '__main__':
    arg_parser = ArgParser()
    args = arg_parser.args
    if not args.net_list:
        main()
