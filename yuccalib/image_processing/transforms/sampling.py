from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform
import numpy as np
from skimage.transform import resize


class DownsampleSegForDS(YuccaTransform):
    """
    """
    def __init__(self, seg_key="seg", factors=(1, 0.5, 0.25, 0.125, 0.0625)):
        self.seg_key = seg_key
        self.factors = factors

    @staticmethod
    def get_params():
        #No parameters to retrieve
        pass

    def __downsample__(self, seg, factors):
        orig_type = seg.dtype
        orig_shape = seg.shape
        downsampled_segs = []
        for factor in factors:
            target_shape = np.array(orig_shape).astype(int)
            for i in range(2, len(orig_shape)):
                target_shape[i] *= factor
            canvas = np.zeros(target_shape)
            for b in range(seg.shape[0]):
                for c in range(seg[b].shape[0]):
                    canvas[b, c] = resize(seg[b, c].astype(float), target_shape[2:], 0, mode="edge",
                                          clip=True, anti_aliasing=False).astype(orig_type)
            downsampled_segs.append(canvas)
        return downsampled_segs

    def __call__(self, **data_dict):
        data_dict[self.seg_key] = self.__downsample__(data_dict[self.seg_key], self.factors)
        return data_dict
