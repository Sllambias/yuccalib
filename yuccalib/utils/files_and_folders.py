import pkgutil
import importlib
import numpy as np
import nibabel as nib
import fileinput
import re
import shutil
import os
import warnings
import yaml
from PIL import Image
from lightning.pytorch.callbacks import BasePredictionWriter
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    subdirs,
    maybe_mkdir_p,
)
from yuccalib.utils.softmax import softmax
from yuccalib.utils.nib_utils import reorient_nib_image

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_yaml(file: str):
    with open(file, "r") as f:
        a = yaml.load(f, Loader=yaml.BaseLoader)
    return a


def replace_in_file(file_path, pattern_replacement):
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:
            for pattern, replacement in pattern_replacement.items():
                line = line.replace(pattern, replacement)
            print(line, end="")


def rename_file_or_dir(file: str, patterns: dict):
    # Patterns is a dict of key, value pairs where keys are the words to replace and values are
    # what to substitute them by. E.g. if patterns = {"foo": "bar"}
    # then the sentence "foo bar" --> "bar bar"
    newfile = file
    for k, v in patterns.items():
        newfile = re.sub(k, v, newfile)
    if os.path.isdir(file):
        if newfile != file:
            shutil.move(file, newfile)
    elif os.path.isfile(file):
        os.rename(file, newfile)


def recursive_rename(folder, patterns_in_file, patterns_in_name):
    """
    Takes a top folder and recursively looks through all subfolders and files.
    For all file contents it will replace patterns_in_file keys with the corresponding values.
    For all file names it will replace the patterns_in_name keys with the corresponding values.

    If patterns_in_file = {"llama": "alpaca", "coffee": "TEA"}
    and patterns_in_name = {"foo": "bar", "Foo": "Bar"}
    and we take the file:

    MyPythonFooScript.py
    ---- (with the following lines of code) ----
    llama = 42
    coffee = 123

    something_else = llama + coffee
    ----

    then we will end up with

    MyPythonBarScript.py
    ----
    alpaca = 42
    TEA = 123

    something_else = alpaca + TEA
    """
    dirs = subdirs(folder)
    files = subfiles(folder)
    for file in files:
        replace_in_file(
            file,
            patterns_in_file,
        )
        rename_file_or_dir(file, patterns_in_name)
    for direc in dirs:
        rename_file_or_dir(direc, patterns_in_name)
        recursive_rename(direc)


def recursive_find_python_class(folder: list, class_name: str, current_module: str):
    """
    Stolen from nnUNet model_restore.py.
    Folder = starting path, e.g. join(yucca.__path__[0], 'preprocessing')
    Trainer_name = e.g. YuccaPreprocessor3D
    Current_module = starting module e.g. 'yucca.preprocessing'
    """
    tr = None
    for _, modname, ispkg in pkgutil.iter_modules(folder):
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for _, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    [join(folder[0], modname)],
                    class_name,
                    current_module=next_current_module,
                )
            if tr is not None:
                break

    return tr


def recursive_find_realpath(path):
    """
    This might produce undesirable results if run on a slurm/batch management user, that does not
    the same permissions as you.
    """
    non_linked_dirs = []
    while path:
        if os.path.islink(path):
            path = os.path.realpath(path)
        path, part = os.path.split(path)
        non_linked_dirs.append(part)
        if path == os.path.sep:
            non_linked_dirs.append(path)
            path = False
    return os.path.join(*non_linked_dirs[::-1])


def save_nifti_from_numpy(pred, outpath, properties, compression=9):
    nib.openers.Opener.default_compresslevel = (
        compression  # slight hacky, but it is what it is
    )
    pred = nib.Nifti1Image(pred, properties["affine"], dtype=np.uint8)
    if properties["reoriented"]:
        pred = reorient_nib_image(
            pred, properties["new_orientation"], properties["original_orientation"]
        )
    pred.set_qform(properties["qform"])
    pred.set_sform(properties["sform"])
    nib.save(
        pred,
        outpath + ".nii.gz",
    )
    del pred


def save_png_from_numpy(pred, outpath, properties, compression=9):
    pred = Image.fromarray(pred)
    pred.save(outpath)
    del pred


def save_txt_from_numpy(pred, outpath, properties):
    np.savetxt(outpath, np.atleast_1d(pred).astype(np.uint8), delimiter=",")
    del pred


def save_prediction_from_logits(
    logits, outpath, properties, save_softmax=False, compression=9
):
    if save_softmax:
        softmax_result = softmax(logits)[0].astype(np.float32)
        np.savez_compressed(
            outpath + ".npz", data=softmax_result, properties=properties
        )
    pred = np.argmax(logits, 1)[0]
    if properties.get("save_format") == "png":
        save_png_from_numpy(pred, outpath, properties)
    if properties.get("save_format") == "txt":
        save_txt_from_numpy(pred, outpath, properties)
    else:
        save_nifti_from_numpy(pred, outpath, properties, compression=compression)


def merge_softmax_from_folders(folders: list, outpath, method="sum"):
    maybe_mkdir_p(outpath)
    cases = subfiles(folders[0], suffix=".npz", join=False)
    for folder in folders:
        assert cases == subfiles(folder, suffix=".npz", join=False), (
            f"Found unexpected cases. "
            f"The following two folders do not contain the same cases: \n"
            f"{folders[0]} \n"
            f"{folder}"
        )

    for case in cases:
        files_for_case = [
            np.load(join(folder, case), allow_pickle=True) for folder in folders
        ]
        properties_for_case = files_for_case[0]["properties"]
        files_for_case = [file["data"].astype(np.float32) for file in files_for_case]

        if method == "sum":
            files_for_case = np.sum(files_for_case, axis=0)

        files_for_case = np.argmax(files_for_case, 0)
        save_nifti_from_numpy(
            files_for_case,
            join(outpath, case[:-4]),
            properties=properties_for_case.item(),
        )

    del files_for_case, properties_for_case


class WritePredictionFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, save_softmax, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_softmax = save_softmax

    def write_on_batch_end(self, trainer, pl_module, data_dict, batch_indices, *args):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["case_id"],
        )
        save_prediction_from_logits(
            logits,
            join(self.output_dir, case_id),
            properties=properties,
            save_softmax=self.save_softmax,
        )


# For backwards compatibility
WriteSegFromLogits = WritePredictionFromLogits
