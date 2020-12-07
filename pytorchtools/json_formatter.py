import ast
import platform
from time import strftime


class JsonFormatter:
    # Dictionary used to store the training results and metadata
    def __init__(self, args):
        self._args = args
        self._json_data = None
        self._train_info = None
        self._format_model()
        self._format_train()

    def _format_model(self):
        model = self._args.model
        dataset = self._args.dataset
        classes = ast.literal_eval(self._args.classes)
        gpu = self._args.gpu
        seed = self._args.seed
        stage = self._args.stage

        self._json_data = {"platform": platform.platform(), "date": strftime("%Y-%m-%d_%H:%M:%S"), "impl": "pytorch",
                     "dataset": dataset, "gpu": gpu, "model": model, "classes": classes, "seed": seed,
                     "stage": stage,
                     }

    def _format_train(self):
        method = self._args.method
        epochs = self._args.epochs
        batch_size = self._args.batch_size

        self._json_data["train_params"] = {"method": method,
                                     "epochs": epochs,
                                     "batch_size": batch_size,
                                     "last_epoch": 0,
                                     "train_time": 0.0
                                     }

    @property
    def json_data(self):
        return self._json_data

    @property
    def train_info(self):
        self._train_info = self._json_data["train_params"]
        return self._train_info
