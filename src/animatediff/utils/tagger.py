# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

import glob
import logging
import os

import cv2
import numpy as np
import onnxruntime
import pandas as pd
from PIL import Image
from tqdm.rich import tqdm

from animatediff.utils.util import prepare_wd14tagger

logger = logging.getLogger(__name__)


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im

def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class Tagger:
    def __init__(self, general_threshold, character_threshold, ignore_tokens, with_confidence, is_danbooru_format,is_cpu):
        prepare_wd14tagger()
#        self.model = onnxruntime.InferenceSession("data/models/WD14tagger/model.onnx", providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        if is_cpu:
            self.model = onnxruntime.InferenceSession("data/models/WD14tagger/model.onnx", providers=['CPUExecutionProvider'])
        else:
            self.model = onnxruntime.InferenceSession("data/models/WD14tagger/model.onnx", providers=['CUDAExecutionProvider'])
        df = pd.read_csv("data/models/WD14tagger/selected_tags.csv")
        self.tag_names = df["name"].tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.ignore_tokens = ignore_tokens
        self.with_confidence = with_confidence
        self.is_danbooru_format = is_danbooru_format

    def __call__(
            self,
            image: Image,
            ):

        _, height, width, _ = self.model.get_inputs()[0].shape

        # Alpha to white
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, probs[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > self.general_threshold]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > self.character_threshold]
        character_res = dict(character_res)

        #logger.info(f"{rating=}")
        #logger.info(f"{general_res=}")
        #logger.info(f"{character_res=}")

        general_res = {k:general_res[k] for k in (general_res.keys() - set(self.ignore_tokens)) }
        character_res = {k:character_res[k] for k in (character_res.keys() - set(self.ignore_tokens)) }

        prompt = ""

        if self.with_confidence:
            prompt = [ f"({i}:{character_res[i]:.2f})" for i in (character_res.keys()) ]
            prompt += [ f"({i}:{general_res[i]:.2f})" for i in (general_res.keys()) ]
        else:
            prompt = [ i for i in (character_res.keys()) ]
            prompt += [ i for i in (general_res.keys()) ]

        prompt = ",".join(prompt)

        if not self.is_danbooru_format:
            prompt = prompt.replace("_", " ")

        #logger.info(f"{prompt=}")
        return prompt


def get_labels(frame_dir, interval, general_threshold, character_threshold, ignore_tokens, with_confidence, is_danbooru_format, is_cpu =False):

    import torch

    result = {}
    if os.path.isdir(frame_dir):
        png_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

        png_map ={}
        for png_path in png_list:
            basename_without_ext = os.path.splitext(os.path.basename(png_path))[0]
            png_map[int(basename_without_ext)] = png_path

        with torch.no_grad():
            tagger = Tagger(general_threshold, character_threshold, ignore_tokens, with_confidence, is_danbooru_format, is_cpu)

            for i in tqdm(range(0, len(png_list), interval ), desc=f"WD14tagger"):
                path = png_map[i]

                #logger.info(f"{path=}")

                result[str(i)] = tagger(
                    image= Image.open(path)
                )

            tagger = None

        torch.cuda.empty_cache()

    return result

