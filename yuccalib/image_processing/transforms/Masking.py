import numpy as np
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform


class Masking(YuccaTransform):
    """
    CURRENTLY NOT IMPLEMENTED
    """
    def __init__(self, data_key="image",
                 mask_ratio: tuple | float = 0.25):
        self.data_key = data_key
        self.mask_ratio = mask_ratio

    @staticmethod
    def get_params(shape, ratio, start_idx):
        pass

    def __mask__(self, image, seg, crop_start_idx):
        pass
    
    def __call__(self, **data_dict):
        return data_dict
