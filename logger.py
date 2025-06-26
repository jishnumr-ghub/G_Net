import os
import sys
import time
import logging
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TextLogger(object):
    """Redirects stdout and stderr to both console and a log file."""
    def __init__(self, log_path):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log = open(log_path, "a")
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal_stdout.flush()
        self.log.flush()

    def close(self):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        self.log.close()


class CompleteLogger:
    """
    CompleteLogger: Console + file logger + optional TensorBoard

    Args:
        root (str): base log directory
        phase (str): "train", "val", etc.
        experiment_name (str): custom experiment identifier
        use_tensorboard (bool): create SummaryWriter or not
    """

    def __init__(self, root="logs", phase="train", experiment_name=None, use_tensorboard=False):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.phase = phase
        self.log_dir = os.path.join(root, experiment_name or phase, now)
        os.makedirs(self.log_dir, exist_ok=True)

        # File path for logging
        self.log_file = os.path.join(self.log_dir, f"{phase}.log")

        # Set up file logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger()
        self.text_logger = TextLogger(self.log_file)

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info("TensorBoard logging enabled.")

    def log_metric(self, tag, value, step):
        """Log scalar to TensorBoard (if enabled)."""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

    def log_config(self, config: dict):
        """Log config settings at start."""
        self.logger.info("🔧 Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def close(self):
        """Close all handlers."""
        self.text_logger.close()
        if self.tb_writer:
            self.tb_writer.close()
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
