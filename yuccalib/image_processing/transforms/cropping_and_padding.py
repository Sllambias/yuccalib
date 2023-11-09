import numpy as np
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform


class CropPad(YuccaTransform):
    """
    The possible input for CropPad can be either 2D or 3D data and the output can be either 2D
    or 3D data. Each dimension is either padded or cropped to the desired patch size.
    For 3D input to 3D output and 2D input to 2D output each dimension is simply
    either padded or cropped to the desired patch size.
    For 3D input to 2D output we extract a slice from the first dimension to attain a 2D image.
    Afterwards the the last 2 dimensions are padded or cropped to the patch size.
    """

    def __init__(
        self,
        data_key="image",
        seg_key="seg",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.seg_key = seg_key
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(input_shape, target_shape):
        target_image_shape = (input_shape[0], *target_shape)
        target_seg_shape = (1, *target_shape)
        return target_image_shape, target_seg_shape

    def __croppad__(
        self,
        image: np.ndarray,
        seg: np.ndarray,
        image_properties: dict,
        target_image_shape: list | tuple,
        target_seg_shape: list | tuple,
    ):
        if len(self.patch_size) == 3:
            return self.generate_3D_case_from_3D(
                image, seg, image_properties, target_image_shape, target_seg_shape
            )
        elif len(self.patch_size) == 2 and len(self.input_shape) == 4:
            return self.generate_2D_case_from_3D(
                image, seg, image_properties, target_image_shape, target_seg_shape
            )
        elif len(self.patch_size) == 2 and len(self.input_shape) == 3:
            return self.generate_2D_case_from_2D(
                image, seg, image_properties, target_image_shape, target_seg_shape
            )

    def generate_3D_case_from_3D(
        self, image, seg, image_properties, target_image_shape, target_seg_shape
    ):
        image_out = np.zeros(target_image_shape)
        seg_out = np.zeros(target_seg_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(3):
            if image.shape[d + 1] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 1])]
            else:
                to_pad += [0]

        pad_lb_x = to_pad[0] // 2
        pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_y = to_pad[1] // 2
        pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2
        pad_lb_z = to_pad[2] // 2
        pad_ub_z = to_pad[2] // 2 + to_pad[2] % 2

        # This is where we should implement any patch selection biases.
        # The final patch excted after augmentation will always be the center of this patch
        # to avoid interpolation artifacts near the borders
        crop_start_idx = []
        if (
            len(image_properties["foreground_locations"]) == 0
            or np.random.uniform() >= self.p_oversample_foreground
        ):
            for d in range(3):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(image.shape[d + 1] - self.patch_size[d] + 1)
                    ]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            for d in range(3):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] - self.patch_size[d]),
                            min(location[d], image.shape[d + 1] - self.patch_size[d])
                            + 1,
                        )
                    ]

        image_out[
            :,
            :,
            :,
            :,
        ] = np.pad(
            image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.patch_size[2],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
            mode="edge",
        )
        if seg is None:
            return image_out, None

        seg_out[
            :,
            :,
            :,
            :,
        ] = np.pad(
            seg[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
                crop_start_idx[2] : crop_start_idx[2] + self.patch_size[2],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        )
        return image_out, seg_out

    def generate_2D_batch_from_3D(
        self, image, seg, image_properties, target_image_shape, target_seg_shape
    ):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_out = np.zeros(target_image_shape)
        seg_out = np.zeros(target_seg_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(2):
            if image.shape[d + 2] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 2])]
            else:
                to_pad += [0]

        pad_lb_y = to_pad[0] // 2
        pad_ub_y = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_z = to_pad[1] // 2
        pad_ub_z = to_pad[1] // 2 + to_pad[1] % 2

        # This is where we should implement any patch selection biases.
        # The final patch extracted after augmentation will always be the center of this patch
        # as this is where augmentation-induced interpolation artefacts are least likely
        crop_start_idx = []
        if (
            len(image_properties["foreground_locations"]) == 0
            or np.random.uniform() >= self.p_oversample_foreground
        ):
            x_idx = np.random.randint(image.shape[1])
            for d in range(2):
                if image.shape[d + 2] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(image.shape[d + 2] - self.patch_size[d] + 1)
                    ]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            x_idx = location[0]
            for d in range(2):
                if image.shape[d + 2] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d + 1] - self.patch_size[d]),
                            min(
                                location[d + 1], image.shape[d + 2] - self.patch_size[d]
                            )
                            + 1,
                        )
                    ]

        image_out[:, :, :, :] = np.pad(
            image[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
            mode="edge",
        )

        if seg is None:
            return image_out, None

        seg_out[:, :, :, :] = np.pad(
            seg[
                :,
                x_idx,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_y, pad_ub_y), (pad_lb_z, pad_ub_z)),
        )

        return image_out, seg_out

    def generate_2D_batch_from_2D(
        self, image, seg, image_properties, target_image_shape, target_seg_shape
    ):
        """
        The possible input for this can be 2D or 3D data.
        For 2D we want to pad or crop as necessary.
        For 3D we want to first select a slice from the first dimension, i.e. volume[idx, :, :],
        then pad or crop as necessary.
        """
        image_out = np.zeros(target_image_shape)
        seg_out = np.zeros(target_seg_shape)

        # First we pad to ensure min size is met
        to_pad = []
        for d in range(2):
            if image.shape[d + 1] < self.patch_size[d]:
                to_pad += [(self.patch_size[d] - image.shape[d + 1])]
            else:
                to_pad += [0]

        pad_lb_x = to_pad[0] // 2
        pad_ub_x = to_pad[0] // 2 + to_pad[0] % 2
        pad_lb_y = to_pad[1] // 2
        pad_ub_y = to_pad[1] // 2 + to_pad[1] % 2

        # This is where we should implement any patch selection biases.
        # The final patch extracted after augmentation will always be the center of this patch
        # as this is where artefacts are least present
        crop_start_idx = []
        if (
            len(image_properties["foreground_locations"]) == 0
            or np.random.uniform() >= self.p_oversample_foreground
        ):
            for d in range(2):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(image.shape[d + 1] - self.patch_size[d] + 1)
                    ]
        else:
            locidx = np.random.choice(len(image_properties["foreground_locations"]))
            location = image_properties["foreground_locations"][locidx]
            for d in range(2):
                if image.shape[d + 1] < self.patch_size[d]:
                    crop_start_idx += [0]
                else:
                    crop_start_idx += [
                        np.random.randint(
                            max(0, location[d] - self.patch_size[d]),
                            min(location[d], image.shape[d + 1] - self.patch_size[d])
                            + 1,
                        )
                    ]

        image_out[:, :, :, :] = np.pad(
            image[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
            mode="edge",
        )

        if seg is None:
            return image_out, None

        seg_out[:, :, :, :] = np.pad(
            seg[
                :,
                crop_start_idx[0] : crop_start_idx[0] + self.patch_size[0],
                crop_start_idx[1] : crop_start_idx[1] + self.patch_size[1],
            ],
            ((0, 0), (pad_lb_x, pad_ub_x), (pad_lb_y, pad_ub_y)),
        )

        return image_out, seg_out

    def __call__(
        self, packed_data_dict=None, image_properties=None, **unpacked_data_dict
    ):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict

        target_image_shape, target_seg_shape = self.get_params(
            data_dict[self.data_key].shape, self.patch_size
        )

        data_dict[self.data_key], data_dict[self.seg_key] = self.__croppad__(
            data_dict[self.data_key],
            data_dict[self.seg_key],
            image_properties,
            target_image_shape,
            target_seg_shape,
        )
        return data_dict
