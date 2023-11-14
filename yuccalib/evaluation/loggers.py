from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from time import localtime, strftime, time
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    isdir,
    subdirs,
)
import sys
import os


class TXTLogger(Logger):
    def __init__(
        self, save_dir: str = "./", name: str = None, steps_per_epoch: int = None
    ):
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self.steps_per_epoch = steps_per_epoch
        self.create_logfile()
        self.previous_epoch = 0
        self.epoch_start_time = time()

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return "0.1"

    @property
    def save_dir(self):
        return self._save_dir

    @rank_zero_only
    def create_logfile(self):
        if not self.name:
            self.name = localtime()
        maybe_mkdir_p(join(self.save_dir, self.name))
        self.log_file = join(
            self.save_dir,
            self.name,
            "training_log.txt",
        )
        with open(self.log_file, "w") as f:
            f.write("Starting model training")
            print("Starting model training \n" f'{"log file:":20} {self.log_file} \n')
            f.write("\n")
            f.write(f'{"log file:":20} {self.log_file}')
            f.write("\n")

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        t = strftime("%Y_%m_%d_%H_%M_%S", localtime())
        with open(self.log_file, "a+") as f:
            current_epoch = (step + 1) // self.steps_per_epoch
            if current_epoch != self.previous_epoch:
                epoch_end_time = time()
                f.write("\n")
                f.write("\n")
                print("\n")
                f.write(f"{t} {'Current Epoch:':20} {current_epoch} \n")
                f.write(
                    f"{t} {'Epoch Time:':20} {epoch_end_time-self.epoch_start_time} \n"
                )
                print(f"{t} {'Current Epoch:':20} {current_epoch}")
                print(f"{t} {'Epoch Time:':20} {epoch_end_time-self.epoch_start_time}")
                self.previous_epoch = current_epoch
                self.epoch_start_time = epoch_end_time
            for key in metrics:
                if key == "epoch":
                    continue
                f.write(f"{t} {key+':':20} {metrics[key]} \n")
                print(f"{t} {key+':':20} {metrics[key]}")
        sys.stdout.flush()

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
