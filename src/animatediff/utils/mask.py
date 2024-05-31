import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
from segment_anything_hq import (SamPredictor, build_sam_vit_b,
                                 build_sam_vit_h, build_sam_vit_l)
from segment_anything_hq.build_sam import build_sam_vit_t
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)

build_sam_table={
    "sam_hq_vit_l":build_sam_vit_l,
    "sam_hq_vit_h":build_sam_vit_h,
    "sam_hq_vit_b":build_sam_vit_b,
    "sam_hq_vit_tiny":build_sam_vit_t,
}

# adapted from https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py
class MaskPredictor:
    def __init__(self,model_config_path, model_checkpoint_path,device, sam_checkpoint, box_threshold=0.3, text_threshold=0.25 ):
        self.groundingdino_model = None
        self.sam_predictor = None

        self.model_config_path = model_config_path
        self.model_checkpoint_path = model_checkpoint_path
        self.device = device
        self.sam_checkpoint = sam_checkpoint

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def load_groundingdino_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        #print(load_res)
        _ = model.eval()
        self.groundingdino_model = model

    def load_sam_predictor(self):
        s = Path(self.sam_checkpoint)
        self.sam_predictor = SamPredictor(build_sam_table[ s.stem ](checkpoint=self.sam_checkpoint).to(self.device))

    def transform_image(self,image_pil):
        import groundingdino.datasets.transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(self, image, caption, with_logits=True):
        model = self.groundingdino_model
        device = self.device

        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases


    def __call__(self, image_pil:Image, text_prompt):
        if self.groundingdino_model is None:
            self.load_groundingdino_model()
            self.load_sam_predictor()

        transformed_img = self.transform_image(image_pil)

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            transformed_img, text_prompt
        )

        if boxes_filt.shape[0] == 0:
            logger.info(f"object not found")
            w, h = image_pil.size
            return np.zeros(shape=(1,h,w), dtype=bool)

        img_array = np.array(image_pil)
        self.sam_predictor.set_image(img_array)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, img_array.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        result = None
        for m in masks:
            if result is None:
                result = m
            else:
                result |= m

        result = result.cpu().detach().numpy().copy()

        return result

def load_mask_list(mask_dir, masked_area_list, mask_padding):

    mask_frame_list = sorted(glob.glob( os.path.join(mask_dir, "[0-9]*.png"), recursive=False))

    kernel = np.ones((abs(mask_padding),abs(mask_padding)),np.uint8)

    for m in mask_frame_list:
        cur = int(Path(m).stem)
        tmp = np.asarray(Image.open(m))

        if mask_padding < 0:
            tmp = cv2.erode(tmp, kernel,iterations = 1)
        elif mask_padding > 0:
            tmp = cv2.dilate(tmp, kernel,iterations = 1)

        masked_area_list[cur] = tmp[None,...]

    return masked_area_list

def crop_mask_list(mask_list):
    area_list = []

    max_h = 0
    max_w = 0

    for m in mask_list:
        if m is None:
            area_list.append(None)
            continue
        m = m > 127
        area = np.where(m[0] == True)
        if area[0].size == 0:
            area_list.append(None)
            continue

        ymin = min(area[0])
        ymax = max(area[0])
        xmin = min(area[1])
        xmax = max(area[1])
        h = ymax+1 - ymin
        w = xmax+1 - xmin
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        area_list.append( (ymin, ymax, xmin, xmax) )
        #crop = m[ymin:ymax+1,xmin:xmax+1]

    logger.info(f"{max_h=}")
    logger.info(f"{max_w=}")

    border_h = mask_list[0].shape[1]
    border_w = mask_list[0].shape[2]

    mask_pos_list=[]
    cropped_mask_list=[]

    for a, m in zip(area_list, mask_list):
        if m is None or a is None:
            mask_pos_list.append(None)
            cropped_mask_list.append(None)
            continue

        ymin,ymax,xmin,xmax = a
        h = ymax+1 - ymin
        w = xmax+1 - xmin

        # H
        diff_h = max_h - h
        dh1 = diff_h//2
        dh2 = diff_h - dh1
        y1 = ymin - dh1
        y2 = ymax + dh2
        if y1 < 0:
            y1 = 0
            y2 = max_h-1
        elif y2 >= border_h:
            y1 = (border_h-1) - (max_h - 1)
            y2 = (border_h-1)

        # W
        diff_w = max_w - w
        dw1 = diff_w//2
        dw2 = diff_w - dw1
        x1 = xmin - dw1
        x2 = xmax + dw2
        if x1 < 0:
            x1 = 0
            x2 = max_w-1
        elif x2 >= border_w:
            x1 = (border_w-1) - (max_w - 1)
            x2 = (border_w-1)

        mask_pos_list.append( (int(x1),int(y1)) )
        m = m[0][y1:y2+1,x1:x2+1]
        cropped_mask_list.append( m[None,...] )


    return cropped_mask_list, mask_pos_list, (max_h,max_w)

def crop_frames(pos_list, crop_size_hw, frame_dir):
    h,w = crop_size_hw

    for i,pos in tqdm(enumerate(pos_list),total=len(pos_list)):
        filename = f"{i:08d}.png"
        frame_path = frame_dir / filename
        if not frame_path.is_file():
            logger.info(f"{frame_path=} not found. skip")
            continue
        if pos is None:
            continue

        x, y = pos

        tmp = np.asarray(Image.open(frame_path))
        tmp = tmp[y:y+h,x:x+w,...]
        Image.fromarray(tmp).save(frame_path)

def save_crop_info(mask_pos_list, crop_size_hw, frame_size_hw, save_path):
    import json

    pos_map = {}

    for i, pos in enumerate(mask_pos_list):
        if pos is not None:
            pos_map[str(i)]=pos

    info = {
        "frame_height" : int(frame_size_hw[0]),
        "frame_width" : int(frame_size_hw[1]),
        "height": int(crop_size_hw[0]),
        "width": int(crop_size_hw[1]),
        "pos_map" : pos_map,
    }

    with open(save_path, mode="wt", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

def restore_position(mask_list, crop_info):

    f_h = crop_info["frame_height"]
    f_w = crop_info["frame_width"]

    h = crop_info["height"]
    w = crop_info["width"]
    pos_map = crop_info["pos_map"]

    for i in pos_map:
        x,y = pos_map[i]
        i = int(i)

        m = mask_list[i]

        if m is None:
            continue

        m = cv2.resize( m, (w,h) )
        if len(m.shape) == 2:
            m = m[...,None]

        frame = np.zeros(shape=(f_h,f_w,m.shape[2]), dtype=np.uint8)

        frame[y:y+h,x:x+w,...] = m
        mask_list[i] = frame


    return mask_list

def load_frame_list(frame_dir, frame_array_list, crop_info):
    frame_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

    for f in frame_list:
        cur = int(Path(f).stem)
        frame_array_list[cur] = np.asarray(Image.open(f))

    if not crop_info:
        logger.info(f"crop_info is not exists -> skip restore")
        return frame_array_list

    for i,f in enumerate(frame_array_list):
        if f is None:
            continue
        frame_array_list[i] = f

    frame_array_list = restore_position(frame_array_list, crop_info)

    return frame_array_list


def create_fg(mask_token, frame_dir, output_dir, output_mask_dir, masked_area_list,
              box_threshold=0.3,
              text_threshold=0.25,
              bg_color=(0,255,0),
              mask_padding=0,
              groundingdino_config="config/GroundingDINO/GroundingDINO_SwinB_cfg.py",
              groundingdino_checkpoint="data/models/GroundingDINO/groundingdino_swinb_cogcoor.pth",
              sam_checkpoint="data/models/SAM/sam_hq_vit_l.pth",
              device="cuda",
              ):

    frame_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

    with torch.no_grad():
        predictor = MaskPredictor(
            model_config_path=groundingdino_config,
            model_checkpoint_path=groundingdino_checkpoint,
            device=device,
            sam_checkpoint=sam_checkpoint,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )


        if mask_padding != 0:
            kernel = np.ones((abs(mask_padding),abs(mask_padding)),np.uint8)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for i, frame in tqdm(enumerate(frame_list),total=len(frame_list), desc=f"creating mask from {mask_token=}"):
            frame = Path(frame)
            file_name = frame.name

            cur_frame_no = int(frame.stem)

            img = Image.open(frame)

            mask_array = predictor(img, mask_token)
            mask_array = mask_array[0].astype(np.uint8) * 255


            if mask_padding < 0:
                mask_array = cv2.erode(mask_array.astype(np.uint8),kernel,iterations = 1)
            elif mask_padding > 0:
                mask_array = cv2.dilate(mask_array.astype(np.uint8),kernel,iterations = 1)

            mask_array = cv2.morphologyEx(mask_array.astype(np.uint8), cv2.MORPH_OPEN, kernel2)
            mask_array = cv2.GaussianBlur(mask_array, (7, 7), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

            if masked_area_list[cur_frame_no] is not None:
                masked_area_list[cur_frame_no] = np.where(masked_area_list[cur_frame_no] > mask_array[None,...], masked_area_list[cur_frame_no], mask_array[None,...])
                #masked_area_list[cur_frame_no] = masked_area_list[cur_frame_no] | mask_array[None,...]
            else:
                masked_area_list[cur_frame_no] = mask_array[None,...]


            if output_mask_dir:
                #mask_array2 = mask_array.astype(np.uint8).clip(0,1)
                #mask_array2 *= 255
                Image.fromarray(mask_array).save( output_mask_dir / file_name )

            img_array = np.asarray(img).copy()
            if bg_color is not None:
                img_array[mask_array == 0] = bg_color

            img = Image.fromarray(img_array)

            img.save( output_dir / file_name )

    return masked_area_list


def dilate_mask(masked_area_list, flow_mask_dilates=8, mask_dilates=5):
    kernel = np.ones((flow_mask_dilates,flow_mask_dilates),np.uint8)
    flow_masks = [ cv2.dilate(mask[0].astype(np.uint8),kernel,iterations = 1) for mask in masked_area_list ]
    flow_masks = [ Image.fromarray(mask * 255) for mask in flow_masks ]

    kernel = np.ones((mask_dilates,mask_dilates),np.uint8)
    dilated_masks = [ cv2.dilate(mask[0].astype(np.uint8),kernel,iterations = 1) for mask in masked_area_list ]
    dilated_masks = [ Image.fromarray(mask * 255) for mask in dilated_masks ]

    return flow_masks, dilated_masks


# adapted from https://github.com/sczhou/ProPainter/blob/main/inference_propainter.py
def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

    return frames, process_size, out_size

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def create_bg(frame_dir, output_dir, masked_area_list,
              use_half = True,
              raft_iter = 20,
              subvideo_length=80,
              neighbor_length=10,
              ref_stride=10,
              device="cuda",
              low_vram = False,
              ):
    import sys
    repo_path = Path("src/animatediff/repo/ProPainter").absolute()
    repo_path = str(repo_path)
    sys.path.append(repo_path)

    from animatediff.repo.ProPainter.core.utils import to_tensors
    from animatediff.repo.ProPainter.model.modules.flow_comp_raft import \
        RAFT_bi
    from animatediff.repo.ProPainter.model.propainter import InpaintGenerator
    from animatediff.repo.ProPainter.model.recurrent_flow_completion import \
        RecurrentFlowCompleteNet
    from animatediff.repo.ProPainter.utils.download_util import \
        load_file_from_url

    pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
    model_dir = Path("data/models/ProPainter")
    model_dir.mkdir(parents=True, exist_ok=True)

    frame_list = sorted(glob.glob( os.path.join(frame_dir, "[0-9]*.png"), recursive=False))

    frames = [Image.open(f) for f in frame_list]

    if low_vram:
        org_size = frames[0].size
        _w, _h = frames[0].size
        if max(_w, _h) > 512:
            _w = int(_w * 0.75)
            _h = int(_h * 0.75)

        frames, size, out_size = resize_frames(frames, (_w, _h))
        out_size = org_size

        masked_area_list = [m[0] for m in masked_area_list]
        masked_area_list = [cv2.resize(m.astype(np.uint8), dsize=size) for m in masked_area_list]
        masked_area_list = [ m>127 for m in masked_area_list]
        masked_area_list = [m[None,...] for m in masked_area_list]

    else:
        frames, size, out_size = resize_frames(frames, None)
        masked_area_list = [ m>127 for m in masked_area_list]

    w, h = size

    flow_masks,masks_dilated = dilate_mask(masked_area_list)

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)


    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'),
                                    model_dir=model_dir, progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
                                    model_dir=model_dir, progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
                                    model_dir=model_dir, progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()



    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    logger.info(f'\nProcessing: [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if max(w,h) <= 640:
            short_clip_len = 12
        elif max(w,h) <= 720:
            short_clip_len = 8
        elif max(w,h) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=raft_iter)
            torch.cuda.empty_cache()


        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()


        # ---- complete flow ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()


        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f],
                                                                       pred_flows_bi_sub,
                                                                       masks_dilated[:, s_f:e_f],
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1

    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        torch.cuda.empty_cache()

    # save each frame
    for idx in range(video_length):
        f = comp_frames[idx]
        f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        dst_img_path = output_dir.joinpath( f"{idx:08d}.png" )
        cv2.imwrite(str(dst_img_path), f)

    sys.path.remove(repo_path)















