from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from typing import Tuple


class Mirror(YuccaTransform):
    """
    variables in DIKU_3D_augmentation_params:
        do_multiplicativeNoise
        multiplicativeNoise_p_per_sample
        multiplicativeNoise_mean
        multiplicativeNoise_sigma
    """

    def __init__(
        self,
        data_key="image",
        seg_key="seg",
        p_per_sample=1,
        axes=(0, 1, 2),
        p_mirror_per_axis=0.33,
        skip_seg=False,
    ):
        self.data_key = data_key
        self.seg_key = seg_key
        self.p_per_sample = p_per_sample
        self.p_mirror_per_axis = p_mirror_per_axis
        self.axes = axes
        self.skip_seg = skip_seg

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __mirror__(self, imageVolume, segVolume, axes):
        # Input will be [c, x, y, z] or [c, x, y]
        if 0 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :] = imageVolume[:, ::-1]
            segVolume[:, :] = segVolume[:, ::-1]
        if 1 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :, :] = imageVolume[:, :, ::-1]
            segVolume[:, :, :] = segVolume[:, :, ::-1]
        if 2 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :, :, :] = imageVolume[:, :, :, ::-1]
            segVolume[:, :, :, :] = segVolume[:, :, :, ::-1]
        return imageVolume, segVolume

    def __mirrorimage__(self, imageVolume, axes):
        # Input will be [c, x, y, z] or [c, x, y]
        if 0 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :] = imageVolume[:, ::-1]
        if 1 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :, :] = imageVolume[:, :, ::-1]
        if 2 in axes and np.random.uniform() < self.p_mirror_per_axis:
            imageVolume[:, :, :, :] = imageVolume[:, :, :, ::-1]
        return imageVolume

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5
            or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"

        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                if self.skip_seg:
                    data_dict[self.data_key][b] = self.__mirrorimage__(
                        data_dict[self.data_key][b], self.axes
                    )
                else:
                    (
                        data_dict[self.data_key][b],
                        data_dict[self.seg_key][b],
                    ) = self.__mirror__(
                        data_dict[self.data_key][b],
                        data_dict[self.seg_key][b],
                        self.axes,
                    )
        return data_dict
