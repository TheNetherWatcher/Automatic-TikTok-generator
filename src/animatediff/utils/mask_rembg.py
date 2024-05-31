import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rembg import new_session, remove
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)

def rembg_create_fg(frame_dir, output_dir, output_mask_dir, masked_area_list,
              bg_color=(0,255,0),
              mask_padding=0,
              ):

    frame_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

    if mask_padding != 0:
        kernel = np.ones((abs(mask_padding),abs(mask_padding)),np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    session = new_session(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    for i, frame in tqdm(enumerate(frame_list),total=len(frame_list), desc=f"creating mask"):
        frame = Path(frame)
        file_name = frame.name

        cur_frame_no = int(frame.stem)

        img = Image.open(frame)
        img_array = np.asarray(img)

        mask_array = remove(img_array, only_mask=True, session=session)

        #mask_array = mask_array[None,...]

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



