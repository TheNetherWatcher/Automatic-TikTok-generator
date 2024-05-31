import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
import torch
from PIL import Image
from rembg import new_session, remove
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)

def animseg_create_fg(frame_dir, output_dir, output_mask_dir, masked_area_list,
              bg_color=(0,255,0),
              mask_padding=0,
              ):

    frame_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

    if mask_padding != 0:
        kernel = np.ones((abs(mask_padding),abs(mask_padding)),np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    rmbg_model = rt.InferenceSession("data/models/anime_seg/isnetis.onnx", providers=providers)

    def get_mask(img, s=1024):
        img = (img / 255).astype(np.float32)
        h, w = h0, w0 = img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = np.zeros([s, s, 3], dtype=np.float32)
        img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        mask = rmbg_model.run(None, {'img': img_input})[0][0]
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        mask = cv2.resize(mask, (w0, h0))
        mask = (mask * 255).astype(np.uint8)
        return mask


    for i, frame in tqdm(enumerate(frame_list),total=len(frame_list), desc=f"creating mask"):
        frame = Path(frame)
        file_name = frame.name

        cur_frame_no = int(frame.stem)

        img = Image.open(frame)
        img_array = np.asarray(img)

        mask_array = get_mask(img_array)

#        Image.fromarray(mask_array).save( output_dir / Path("raw_" + file_name))

        if mask_padding < 0:
            mask_array = cv2.erode(mask_array.astype(np.uint8),kernel,iterations = 1)
        elif mask_padding > 0:
            mask_array = cv2.dilate(mask_array.astype(np.uint8),kernel,iterations = 1)

        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel2)
        mask_array = cv2.GaussianBlur(mask_array, (7, 7), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

        if masked_area_list[cur_frame_no] is not None:
            masked_area_list[cur_frame_no] = np.where(masked_area_list[cur_frame_no] > mask_array[None,...], masked_area_list[cur_frame_no], mask_array[None,...])
        else:
            masked_area_list[cur_frame_no] = mask_array[None,...]

        if output_mask_dir:
            Image.fromarray(mask_array).save( output_mask_dir / file_name )

        img_array = np.asarray(img).copy()
        if bg_color is not None:
            img_array[mask_array == 0] = bg_color

        img = Image.fromarray(img_array)

        img.save( output_dir / file_name )

    return masked_area_list



