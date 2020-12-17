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

