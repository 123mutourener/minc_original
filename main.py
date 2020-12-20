from pytorchtools.loop_control.train_loop import Trainer
from pytorchtools.unit_test.test_model import LitMNIST

model = LitMNIST()
trainer = Trainer(max_epochs=2, progress_bar_refresh_rate=20)
trainer.fit(model)