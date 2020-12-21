from pytorchtools.loop_control.train_loop import Trainer
from pytorchtools.unit_test.test_model import LitMNIST
from pytorchtools.callbacks import valid_acc_callback, valid_loss_callback, last_callback

tag = "unit_test"

valid_loss_callback = valid_loss_callback(tag=tag)
valid_acc_callback = valid_acc_callback(tag=tag)
last_callback = last_callback(tag=tag)

model = LitMNIST()
trainer = Trainer(max_epochs=3, progress_bar_refresh_rate=20, callbacks=[valid_loss_callback, valid_acc_callback, last_callback],
                  # resume_from_checkpoint="./checkpoints/"+tag+"/accuracy/minc-epoch=01-valid_accuracy=-0.48.ckpt",
                  resume_from_checkpoint=None
                  )
trainer.fit(model)
