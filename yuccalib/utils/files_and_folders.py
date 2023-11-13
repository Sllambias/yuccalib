import pkgutil
import importlib
import numpy as np
import nibabel as nib
from lightning.pytorch.callbacks import BasePredictionWriter
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    subfiles,
    maybe_mkdir_p,
)
from yuccalib.utils.softmax import softmax
from yuccalib.utils.nib_utils import reorient_nib_image
import os
import warnings
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_yaml(file: str):
    with open(file, "r") as f:
        a = yaml.load(f, Loader=yaml.BaseLoader)
    return a


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


def save_segmentation_from_numpy(seg, outpath, properties, compression=9):
    nib.openers.Opener.default_compresslevel = (
        compression  # slight hacky, but it is what it is
    )
    seg = nib.Nifti1Image(seg, properties["affine"], dtype=np.uint8)
    if properties["reoriented"]:
        seg = reorient_nib_image(
            seg, properties["new_orientation"], properties["original_orientation"]
        )
    seg.set_qform(properties["qform"])
    seg.set_sform(properties["sform"])
    nib.save(
        seg,
        outpath + ".nii.gz",
    )
    del seg


def save_segmentation_from_logits(
    logits, outpath, properties, save_softmax=False, compression=9
):
    if save_softmax:
        softmax_result = softmax(logits)[0].astype(np.float32)
        np.savez_compressed(
            outpath + ".npz", data=softmax_result, properties=properties
        )
    seg = np.argmax(logits, 1)[0]
    save_segmentation_from_numpy(seg, outpath, properties, compression=compression)


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
        save_segmentation_from_numpy(
            files_for_case,
            join(outpath, case[:-4]),
            properties=properties_for_case.item(),
        )

    del files_for_case, properties_for_case


class WriteSegFromLogits(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, data_dict, batch_indices, *args):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        logits, properties, case_id = (
            data_dict["logits"],
            data_dict["properties"],
            data_dict["case_id"],
        )
        save_segmentation_from_logits(
            logits, join(self.output_dir, case_id), properties=properties
        )
