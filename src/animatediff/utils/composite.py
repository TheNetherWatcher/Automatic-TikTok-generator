import glob
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)


#https://github.com/jinwonkim93/laplacian-pyramid-blend
#https://blog.shikoan.com/pytorch-laplacian-pyramid/
class LaplacianPyramidBlender:

	device = None

	def get_gaussian_kernel(self):
		kernel = np.array([
			[1, 4, 6, 4, 1],
			[4, 16, 24, 16, 4],
			[6, 24, 36, 24, 6],
			[4, 16, 24, 16, 4],
			[1, 4, 6, 4, 1]], np.float32) / 256.0
		gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5),device=self.device)
		return gaussian_k

	def pyramid_down(self, image):
		with torch.no_grad():
			gaussian_k = self.get_gaussian_kernel()
			multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2) for i in range(3)]
			down_image = torch.cat(multiband, dim=1)
		return down_image

	def pyramid_up(self, image, size = None):
		with torch.no_grad():
			gaussian_k = self.get_gaussian_kernel()
			if size is None:
				upsample = F.interpolate(image, scale_factor=2)
			else:
				upsample = F.interpolate(image, size=size)
			multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2) for i in range(3)]
			up_image = torch.cat(multiband, dim=1)
		return up_image

	def gaussian_pyramid(self, original, n_pyramids):
		x = original
		# pyramid down
		pyramids = [original]
		for i in range(n_pyramids):
			x = self.pyramid_down(x)
			pyramids.append(x)
		return pyramids

	def laplacian_pyramid(self, original, n_pyramids):
		pyramids = self.gaussian_pyramid(original, n_pyramids)

		# pyramid up - diff
		laplacian = []
		for i in range(len(pyramids) - 1):
			diff = pyramids[i] - self.pyramid_up(pyramids[i + 1], pyramids[i].shape[2:])
			laplacian.append(diff)

		laplacian.append(pyramids[-1])
		return laplacian

	def laplacian_pyramid_blending_with_mask(self, src, target, mask, num_levels = 9):
        # assume mask is float32 [0,1]

		# generate Gaussian pyramid for src,target and mask

		Gsrc = torch.as_tensor(np.expand_dims(src, axis=0), device=self.device)
		Gtarget = torch.as_tensor(np.expand_dims(target, axis=0), device=self.device)
		Gmask = torch.as_tensor(np.expand_dims(mask, axis=0), device=self.device)

		lpA = self.laplacian_pyramid(Gsrc,num_levels)[::-1]
		lpB = self.laplacian_pyramid(Gtarget,num_levels)[::-1]
		gpMr = self.gaussian_pyramid(Gmask,num_levels)[::-1]

		# Now blend images according to mask in each level
		LS = []
		for idx, (la,lb,Gmask) in enumerate(zip(lpA,lpB,gpMr)):
			lo = lb * (1.0 - Gmask)
			if idx <= 2:
				lo += lb * Gmask
			else:
				lo +=  la * Gmask
			LS.append(lo)

		# now reconstruct
		ls_ = LS.pop(0)
		for lap in LS:
			ls_ = self.pyramid_up(ls_, lap.shape[2:]) + lap

		result = ls_.squeeze(dim=0).to('cpu').detach().numpy().copy()

		return result

	def __call__(self,
					src_image: np.ndarray,
					target_image: np.ndarray,
					mask_image: np.ndarray,
					device
					):

		self.device = device

		num_levels = int(np.log2(src_image.shape[0]))
		#normalize image to 0, 1
		mask_image = np.clip(mask_image, 0, 1).transpose([2, 0, 1])

		src_image = src_image.transpose([2, 0, 1]).astype(np.float32) / 255.0
		target_image = target_image.transpose([2, 0, 1]).astype(np.float32) / 255.0
		composite_image = self.laplacian_pyramid_blending_with_mask(src_image, target_image, mask_image, num_levels)
		composite_image = np.clip(composite_image*255, 0 , 255).astype(np.uint8)
		composite_image=composite_image.transpose([1, 2, 0])
		return composite_image


def composite(bg_dir, fg_list, output_dir, masked_area_list, device="cuda"):
	bg_list = sorted(glob.glob( os.path.join(bg_dir ,"[0-9]*.png"), recursive=False))

	blender = LaplacianPyramidBlender()

	for bg, fg_array, mask in tqdm(zip(bg_list, fg_list, masked_area_list),total=len(bg_list), desc="compositing"):
		name = Path(bg).name
		save_path = output_dir / name

		if fg_array is None:
			logger.info(f"composite fg_array is None -> skip")
			shutil.copy(bg, save_path)
			continue

		if mask is None:
			logger.info(f"mask is None -> skip")
			shutil.copy(bg, save_path)
			continue

		bg = np.asarray(Image.open(bg)).copy()
		fg = fg_array
		mask = np.concatenate([mask, mask, mask], 2)

		h, w, _ = bg.shape

		fg = cv2.resize(fg, dsize=(w,h))
		mask = cv2.resize(mask, dsize=(w,h))


		mask = mask.astype(np.float32)
#		mask = mask * 255
		mask = cv2.GaussianBlur(mask, (15, 15), 0)
		mask = mask / 255

		fg = fg * mask + bg * (1-mask)

		img = blender(fg, bg, mask,device)


		img = Image.fromarray(img)
		img.save(save_path)

def simple_composite(bg_dir, fg_list, output_dir, masked_area_list, device="cuda"):
	bg_list = sorted(glob.glob( os.path.join(bg_dir ,"[0-9]*.png"), recursive=False))

	for bg, fg_array, mask in tqdm(zip(bg_list, fg_list, masked_area_list),total=len(bg_list), desc="compositing"):
		name = Path(bg).name
		save_path = output_dir / name

		if fg_array is None:
			logger.info(f"composite fg_array is None -> skip")
			shutil.copy(bg, save_path)
			continue

		if mask is None:
			logger.info(f"mask is None -> skip")
			shutil.copy(bg, save_path)
			continue

		bg = np.asarray(Image.open(bg)).copy()
		fg = fg_array
		mask = np.concatenate([mask, mask, mask], 2)

		h, w, _ = bg.shape

		fg = cv2.resize(fg, dsize=(w,h))
		mask = cv2.resize(mask, dsize=(w,h))


		mask = mask.astype(np.float32)
		mask = cv2.GaussianBlur(mask, (15, 15), 0)
		mask = mask / 255

		img = fg * mask + bg * (1-mask)
		img = img.clip(0 , 255).astype(np.uint8)

		img = Image.fromarray(img)
		img.save(save_path)