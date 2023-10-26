"""
https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
From: https://stackoverflow.com/questions/59738230/apply-rotation-defined-by-euler-angles-to-3d-image-in-python
"""
import numpy as np
from scipy.ndimage import map_coordinates
from yuccalib.image_processing.matrix_ops import create_zero_centered_coordinate_matrix, \
	deform_coordinate_matrix, Rx, Ry, Rz, Rz2D
from yuccalib.image_processing.transforms.YuccaTransform import YuccaTransform
from typing import Tuple


class Spatial(YuccaTransform):
	"""
	variables in aug_params:
		do_Rotation
		Rotation_p_per_sample
		Rotation_p_per_channel
		Rotation_x_rot
		Rotation_y_rot
		Rotation_z_rot 
	"""
	def __init__(self, data_key="image", seg_key="seg",
	      			crop=False,
	      			patch_size: Tuple[int] = None,
				    random_crop = True,
					p_deform_per_sample= 1,
					deform_sigma = (20, 30),
					deform_alpha = (300, 600),
					p_rot_per_sample= 1,
					p_rot_per_axis = 1,
					x_rot_in_degrees = (0., 10.),
					y_rot_in_degrees = (0., 10.),
					z_rot_in_degrees = (0., 10.),
					p_scale_per_sample = 1,
					scale_factor = (0.85, 1.15),
					skip_seg=False,
					):
		
		self.data_key = data_key
		self.seg_key = seg_key
		self.skip_seg = skip_seg
		self.do_crop = crop
		self.patch_size = patch_size
		self.random_crop = random_crop

		self.p_deform_per_sample = p_deform_per_sample
		self.deform_sigma = deform_sigma
		self.deform_alpha = deform_alpha

		self.p_rot_per_sample = p_rot_per_sample
		self.p_rot_per_axis = p_rot_per_axis
		self.x_rot_in_degrees = x_rot_in_degrees
		self.y_rot_in_degrees = y_rot_in_degrees
		self.z_rot_in_degrees = z_rot_in_degrees

		self.p_scale_per_sample = p_scale_per_sample
		self.scale_factor = scale_factor

	@staticmethod
	def get_params(deform_alpha: Tuple[float],
	       			deform_sigma: Tuple[float],
					x_rot: Tuple[float],
					y_rot: Tuple[float],
					z_rot: Tuple[float],
					scale_factor: Tuple[float]
					) -> Tuple[float]:
		
		if deform_alpha:
			deform_alpha = float(np.random.uniform(*deform_alpha))
		if deform_sigma:
			deform_sigma = float(np.random.uniform(*deform_sigma))

		if x_rot:
			x_rot = float(np.random.uniform(*x_rot)) * (np.pi/180)
		if y_rot:
			y_rot = float(np.random.uniform(*y_rot)) * (np.pi/180)
		if z_rot:
			z_rot = float(np.random.uniform(*z_rot)) * (np.pi/180)

		if scale_factor:
			scale_factor = float(np.random.uniform(*scale_factor))

		return deform_alpha, deform_sigma, x_rot, y_rot, z_rot, scale_factor


	def __CropDeformRotateScale__(self, imageVolume, segVolume, patch_size, alpha, sigma, 
							   x_rot, y_rot, z_rot, scale_factor,
							   skip_seg):
		if not self.do_crop:
			patch_size = imageVolume.shape[2:]
		
		coords = create_zero_centered_coordinate_matrix(patch_size)
		imageCanvas = np.zeros((imageVolume.shape[0], imageVolume.shape[1], *patch_size), dtype=np.float32)

		# First we apply deformation to the coordinate matrix
		if np.random.uniform() < self.p_deform_per_sample:
			coords = deform_coordinate_matrix(coords, alpha = alpha, sigma = sigma)
	
		# Then we rotate the coordinate matrix around one or more axes
		if np.random.uniform() < self.p_rot_per_sample:
			rot_matrix = np.eye(len(patch_size))
			if len(patch_size) == 2:
				rot_matrix = np.dot(rot_matrix, Rz2D(z_rot))
			else:
				if np.random.uniform() < self.p_rot_per_axis:
					rot_matrix = np.dot(rot_matrix, Rx(x_rot))
				if np.random.uniform() < self.p_rot_per_axis:
					rot_matrix = np.dot(rot_matrix, Ry(y_rot))
				if np.random.uniform() < self.p_rot_per_axis:
					rot_matrix = np.dot(rot_matrix, Rz(z_rot))

			coords = np.dot(coords.reshape(len(patch_size), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)

		# And finally scale it
		# Scaling effect is "inverted"
		# i.e. a scale factor of 0.9 will zoom in
		if np.random.uniform() < self.p_scale_per_sample:
			coords *= scale_factor

		if self.random_crop and self.do_crop:
			for d in range(len(patch_size)):
				crop_center_idx = [np.random.randint(int(patch_size[d]/2),
					  				imageVolume.shape[d+2] - int(patch_size[d]/2) + 1)]
				coords[d] += crop_center_idx
		else:
			# Reversing the zero-centering of the coordinates
			for d in range(len(patch_size)):
				coords[d] += imageVolume.shape[d+2] / 2. - 0.5

		# Mapping the images to the distorted coordinates
		for b in range(imageVolume.shape[0]):
			for c in range(imageVolume.shape[1]):
				imageCanvas[b, c] = map_coordinates(imageVolume[b, c].astype(float),
													coords, order=3,
													mode='constant',
													cval=0.0).astype(imageVolume.dtype)

		if not skip_seg:
			segCanvas = np.zeros((segVolume.shape[0], segVolume.shape[1], *patch_size), dtype=np.float32)

			# Mapping the segmentations to the distorted coordinates
			for b in range(segVolume.shape[0]):
				for c in range(segVolume.shape[1]):
					segCanvas[b, c] = map_coordinates(segVolume[b, c], coords, order=0,
						mode='constant', cval=0.0).astype(segVolume.dtype)
			return imageCanvas, segCanvas
		return imageCanvas, segVolume

	def __call__(self, **data_dict):
			assert (len(data_dict[self.data_key].shape) == 5 or len(data_dict[self.data_key].shape) == 4), f"Incorrect data size or shape.\
				\nShould be (c, x, y, z) or (c, x, y) and is: {data_dict[self.data_key].shape}"

			deform_alpha, deform_sigma, x_rot_rad, y_rot_rad, \
			z_rot_rad, scale_factor = self.get_params(deform_alpha = self.deform_alpha, 
													  deform_sigma = self.deform_sigma,
													  x_rot = self.x_rot_in_degrees, 
													  y_rot = self.y_rot_in_degrees,
													  z_rot = self.z_rot_in_degrees, 
													  scale_factor = self.scale_factor)

			data_dict[self.data_key], data_dict[self.seg_key] = self.__CropDeformRotateScale__(
							data_dict[self.data_key], data_dict[self.seg_key], self.patch_size,
							deform_alpha, deform_sigma,
							x_rot_rad, y_rot_rad, z_rot_rad,
							scale_factor, self.skip_seg)
			return data_dict
