from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from time import localtime, strftime, time
from batchgenerators.utilities.file_and_folder_operations import join
import sys


class TXTLogger(Logger):
    def __init__(self, savedir, steps_per_epoch):
        super().__init__()
        self.savedir = savedir
        self.steps_per_epoch = steps_per_epoch
        self.create_logfile()
        self.previous_epoch = 0
        self.epoch_start_time = time()

    @property
    def name(self):
        return "TXTLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def create_logfile(self):
        initial_time_obj = localtime()
        self.log_file = join(
            self.savedir,
            "log_" + strftime("%Y_%m_%d_%H_%M_%S", initial_time_obj) + ".txt",
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
                f.write(f"{t} {'Current Epoch:':20} {current_epoch}")
                f.write(
                    f"{t} {'Epoch Time:':20} {epoch_end_time-self.epoch_start_time}"
                )
                print(f"{t} {'Current Epoch:':20} {current_epoch}")
                self.previous_epoch = current_epoch
                self.epoch_start_time = epoch_end_time
                f.write("\n")
            for key in metrics:
                f.write(f"{t} {key+':':20} {metrics[key]}")
                print(f"{t} {key+':':20} {metrics[key]}")
                f.write("\n")
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
