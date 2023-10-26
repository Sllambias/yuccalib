import numpy as np
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform


class CropPad(YuccaTransform):
    """
    Wrapper class for torchio.CropOrPad
    """
    def __init__(self, data_key="image", seg_key="seg",
                 patch_size: tuple | list = None,
                 random_crop=False):
        self.data_key = data_key
        self.seg_key = seg_key
        self.patch_size = patch_size
        self.random_crop = random_crop

    @staticmethod
    def get_params(shape, target_shape, random_crop=True):
        if random_crop:
            crop_start_idx = []
            for d in range(len(target_shape)):
                crop_start_idx += [np.random.randint(shape[d] - target_shape[d] + 1)]
        else:
            print("non-random cropping not implemented yet")
        return crop_start_idx

    def __croppad__(self, image, seg, crop_start_idx):
        if len(crop_start_idx) == 3:
            image = image[:, crop_start_idx[0]: crop_start_idx[0] + self.patch_size[0],
                        crop_start_idx[1]: crop_start_idx[1] + self.patch_size[1],
                        crop_start_idx[2]: crop_start_idx[2] + self.patch_size[2]]
            seg = seg[:, crop_start_idx[0]: crop_start_idx[0] + self.patch_size[0],
                    crop_start_idx[1]: crop_start_idx[1] + self.patch_size[1],
                    crop_start_idx[2]: crop_start_idx[2] + self.patch_size[2]]
        if len(crop_start_idx) == 2:
            image = image[:, crop_start_idx[0]: crop_start_idx[0] + self.patch_size[0],
                        crop_start_idx[1]: crop_start_idx[1] + self.patch_size[1]]
            seg = seg[:, crop_start_idx[0]: crop_start_idx[0] + self.patch_size[0],
                    crop_start_idx[1]: crop_start_idx[1] + self.patch_size[1]]
        return image, seg

    def __call__(self, **data_dict):
        data_dict_cropped = {"image": np.zeros((data_dict[self.data_key].shape[0],
                                               data_dict[self.data_key].shape[1],
                                               *self.patch_size)),
                             "seg": np.zeros((data_dict[self.seg_key].shape[0],
                                              data_dict[self.seg_key].shape[1],
                                              *self.patch_size))}

        for b in range(data_dict[self.data_key].shape[0]):

            crop_start_idx = self.get_params(data_dict[self.data_key].shape[2:],
                                             self.patch_size,
                                             self.random_crop)

            data_dict_cropped[self.data_key][b], \
                data_dict_cropped[self.seg_key][b] = self.__croppad__(
                data_dict[self.data_key][b], data_dict[self.seg_key][b], crop_start_idx)

        del data_dict
        return data_dict_cropped
