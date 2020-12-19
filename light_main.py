from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
import os
from pytorchtools.MINCDataModule import MINCDataModule
from archive.arg_parser import ArgParser
from pytorchtools.patch_classifier import PatchClassifier
from pytorchtools.callbacks import valid_acc_callback, valid_loss_callback, last_callback
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# import warnings


def main():
    # warnings.filterwarnings('ignore')
    # Start training from scratch
    seed_everything(args.seed)
    logger = TensorBoardLogger("./lightning_logs", name=args.tag)

    # Parse the argements
    json_data = arg_parser.json_data
    model = PatchClassifier(json_data)
    dm = MINCDataModule(args.data_root, json_data)

    # resume
    if args.resume:
        resume_path = os.path.join(os.getcwd(), "checkpoints", args.tag, "last", "last.ckpt")
    else:
        resume_path = None

    # for HPC training
    if not args.debug:
        trainer = Trainer(progress_bar_refresh_rate=20, log_every_n_steps=20, flush_logs_every_n_steps=800,
                          max_epochs=args.epochs,
                          gpus=args.gpus,
                          num_nodes=args.num_nodes,
                          accelerator='ddp',
                          replace_sampler_ddp=False,
                          callbacks=[valid_acc_callback(args.tag), valid_loss_callback(args.tag), last_callback(args.tag)],
                          logger=logger, resume_from_checkpoint=resume_path)
    else:
        # for CPU training
        trainer = Trainer(progress_bar_refresh_rate=1, log_every_n_steps=1, flush_logs_every_n_steps=1,
                          max_epochs=3, replace_sampler_ddp=False, accelerator='ddp_cpu', num_processes=2,
                          callbacks=[valid_acc_callback(args.tag), valid_loss_callback(args.tag), last_callback(args.tag)],
                          logger=logger, resume_from_checkpoint=resume_path)

    if not args.test:
        trainer.fit(model, dm)

    # run test set
    if args.test:
        # TODO: Write test code in separate file.
        # TODO: Load one of the the saved checkpoints, then test.
        # TODO: See https://pytorch-lightning.readthedocs.io/en/latest/trainer.html?highlight=test#test
        trainer.test()


if __name__ == '__main__':
    arg_parser = ArgParser()
    args = arg_parser.args
    if not args.net_list:
        main()
