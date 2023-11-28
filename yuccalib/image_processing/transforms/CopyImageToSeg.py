from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple


class CopyImageToSeg(YuccaTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_multiplicativeNoise
        multiplicativeNoise_p_per_sample
        multiplicativeNoise_mean
        multiplicativeNoise_sigma
    """

    def __init__(self, data_key="image", label_key="label"):
        self.data_key = data_key
        self.label_key = label_key

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __copy__(self, imageVolume):
        return imageVolume, imageVolume.copy()

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5
            or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        data_dict[self.data_key], data_dict[self.label_key] = self.__copy__(
            data_dict[self.data_key]
        )
        return data_dict
