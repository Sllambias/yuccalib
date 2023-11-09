import numpy as np
import torch
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform


class RemoveSegChannelAxis(YuccaTransform):
    """
    Basically a wrapper for np.squeeze
    """

    def __init__(self, seg_key="seg", channel_to_remove=1):
        self.seg_key = seg_key
        self.channel = channel_to_remove

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __remove_channel__(self, image, channel):
        return np.squeeze(image, axis=channel)

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            data_dict[self.seg_key].shape[self.channel] == 1
        ), f"Invalid operation: attempting to remove channel of size > 1.\
            \nTrying to squeeze channel: {self.channel} of array with shape: {data_dict[self.seg_key].shape}"
        data_dict[self.seg_key] = self.__remove_channel__(
            data_dict[self.seg_key], self.channel
        )
        return data_dict


class NumpyToTorch(YuccaTransform):
    def __init__(self, data_key="image", seg_key="seg", seg_dtype="int"):
        self.data_key = data_key
        self.seg_key = seg_key
        self.seg_dtype = seg_dtype

    def get_params(self):
        if self.seg_dtype == "int":
            self.seg_dtype = torch.int32
        elif self.seg_dtype == "float":
            self.seg_dtype = torch.float32

    def __convert__(self, data, seg):
        data = torch.tensor(data, dtype=torch.float32)
        if isinstance(seg, list):
            seg = [torch.tensor(i, dtype=self.seg_dtype) for i in seg]
        else:
            seg = torch.tensor(seg, dtype=self.seg_dtype)
        return data, seg

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        assert (
            len(data_dict[self.data_key].shape) == 5
            or len(data_dict[self.data_key].shape) == 4
        ), f"Incorrect data size or shape.\
            \nShould be (b, c, x, y, z) or (b, c, x, y) and is: {data_dict[self.data_key].shape}"
        self.get_params()
        data_dict[self.data_key], data_dict[self.seg_key] = self.__convert__(
            data_dict[self.data_key], data_dict[self.seg_key]
        )
        return data_dict


class AddBatchDimension(YuccaTransform):
    def __init__(self, data_key="image", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key

    @staticmethod
    def get_params():
        pass

    def __unsqueeze__(self, data, seg):
        data = data[np.newaxis]
        if isinstance(seg, list):
            seg = [s[np.newaxis] for i in seg]
        else:
            seg = seg[np.newaxis]
        return data, seg

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_dict[self.data_key], data_dict[self.seg_key] = self.__unsqueeze__(
            data_dict[self.data_key], data_dict[self.seg_key]
        )
        return data_dict


class RemoveBatchDimension(YuccaTransform):
    def __init__(self, data_key="image", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key

    @staticmethod
    def get_params():
        pass

    def __squeeze__(self, data, seg):
        data = data[0]
        if isinstance(seg, list):
            seg = [s[0] for i in seg]
        else:
            seg = seg[0]
        return data, seg

    def __call__(self, packed_data_dict=None, **unpacked_data_dict):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        data_dict[self.data_key], data_dict[self.seg_key] = self.__squeeze__(
            data_dict[self.data_key], data_dict[self.seg_key]
        )
        return data_dict
