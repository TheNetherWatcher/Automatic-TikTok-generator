import logging

from safetensors.torch import load_file

from animatediff import get_dir
from animatediff.utils.lora_diffusers import (LoRANetwork,
                                              create_network_from_weights)

logger = logging.getLogger(__name__)

data_dir = get_dir("data")


def merge_safetensors_lora(text_encoder, unet, lora_path, alpha=0.75, is_animatediff=True):

    def dump(loaded):
        for a in loaded:
            logger.info(f"{a} {loaded[a].shape}")

    sd = load_file(lora_path)

    if False:
        dump(sd)

    print(f"create LoRA network")
    lora_network: LoRANetwork = create_network_from_weights(text_encoder, unet, sd, multiplier=alpha, is_animatediff=is_animatediff)
    print(f"load LoRA network weights")
    lora_network.load_state_dict(sd, False)
    lora_network.merge_to(alpha)

def load_lora_map(pipe, lora_map_config, video_length, is_sdxl=False):
    new_map = {}
    for item in lora_map_config:
        lora_path = data_dir.joinpath(item)
        if type(lora_map_config[item]) in (float,int):
            te_en = [pipe.text_encoder, pipe.text_encoder_2] if is_sdxl else pipe.text_encoder
            merge_safetensors_lora(te_en, pipe.unet, lora_path, lora_map_config[item], not is_sdxl)
        else:
            new_map[lora_path] = lora_map_config[item]

    lora_map = LoraMap(pipe, new_map, video_length, is_sdxl)
    pipe.lora_map = lora_map if lora_map.is_valid else None

def load_lcm_lora(pipe, lcm_map, is_sdxl=False, is_merge=False):
    if is_sdxl:
        lora_path = data_dir.joinpath("models/lcm_lora/sdxl/pytorch_lora_weights.safetensors")
    else:
        lora_path = data_dir.joinpath("models/lcm_lora/sd15/pytorch_lora_weights.safetensors")
    logger.info(f"{lora_path=}")

    if is_merge:
        te_en = [pipe.text_encoder, pipe.text_encoder_2] if is_sdxl else pipe.text_encoder
        merge_safetensors_lora(te_en, pipe.unet, lora_path, 1.0, not is_sdxl)
        pipe.lcm = None
        return

    lcm = LcmLora(pipe, is_sdxl, lora_path, lcm_map)
    pipe.lcm = lcm if lcm.is_valid else None

class LcmLora:
    def __init__(
            self,
            pipe,
            is_sdxl,
            lora_path,
            lcm_map
        ):
            self.is_valid = False

            sd = load_file(lora_path)
            if not sd:
                return

            te_en = [pipe.text_encoder, pipe.text_encoder_2] if is_sdxl else pipe.text_encoder
            lora_network: LoRANetwork = create_network_from_weights(te_en, pipe.unet, sd, multiplier=1.0, is_animatediff=not is_sdxl)
            lora_network.load_state_dict(sd, False)
            lora_network.apply_to(1.0)
            self.network = lora_network

            self.is_valid = True

            self.start_scale = lcm_map["start_scale"]
            self.end_scale = lcm_map["end_scale"]
            self.gradient_start = lcm_map["gradient_start"]
            self.gradient_end = lcm_map["gradient_end"]


    def to(
            self,
            device,
            dtype,
        ):
        self.network.to(device=device, dtype=dtype)

    def apply(
            self,
            step,
            total_steps,
        ):
        step += 1
        progress = step / total_steps

        if progress < self.gradient_start:
            scale = self.start_scale
        elif progress > self.gradient_end:
            scale = self.end_scale
        else:
            if (self.gradient_end - self.gradient_start) < 1e-4:
                progress = 0
            else:
                progress = (progress - self.gradient_start) / (self.gradient_end - self.gradient_start)
            scale = (self.end_scale - self.start_scale) * progress
            scale += self.start_scale

        self.network.active( scale )

    def unapply(
            self,
        ):
        self.network.deactive( )



class LoraMap:
    def __init__(
            self,
            pipe,
            lora_map,
            video_length,
            is_sdxl,
        ):
        self.networks = []

        def create_schedule(scales, length):
            scales = { int(i):scales[i] for i in scales }
            keys = sorted(scales.keys())

            if len(keys) == 1:
                return { i:scales[keys[0]] for i in range(length) }
            keys = keys + [keys[0]]

            schedule={}

            def calc(rate,start_v,end_v):
                return start_v + (rate * rate)*(end_v - start_v)

            for key_prev,key_next in zip(keys[:-1],keys[1:]):
                v1 = scales[key_prev]
                v2 = scales[key_next]
                if key_prev > key_next:
                    key_next += length
                for i in range(key_prev,key_next):
                    dist = i-key_prev
                    if i >= length:
                        i -= length
                    schedule[i] = calc( dist/(key_next-key_prev), v1, v2 )
            return schedule

        for lora_path in lora_map:
            sd = load_file(lora_path)
            if not sd:
                continue
            te_en = [pipe.text_encoder, pipe.text_encoder_2] if is_sdxl else pipe.text_encoder
            lora_network: LoRANetwork = create_network_from_weights(te_en, pipe.unet, sd, multiplier=0.75, is_animatediff=not is_sdxl)
            lora_network.load_state_dict(sd, False)
            lora_network.apply_to(0.75)

            self.networks.append(
                {
                    "network":lora_network,
                    "region":lora_map[lora_path]["region"],
                    "schedule": create_schedule(lora_map[lora_path]["scale"], video_length )
                }
            )

        def region_convert(i):
            if i == "background":
                return 0
            else:
                return int(i) + 1

        for net in self.networks:
            net["region"] = [ region_convert(i) for i in net["region"] ]

#        for n in self.networks:
#            logger.info(f"{n['region']=}")
#            logger.info(f"{n['schedule']=}")

        if self.networks:
            self.is_valid = True
        else:
            self.is_valid = False

    def to(
            self,
            device,
            dtype,
        ):
        for net in self.networks:
            net["network"].to(device=device, dtype=dtype)

    def apply(
            self,
            cond_index,
            cond_nums,
            frame_no,
        ):
        '''
        neg 0 (bg)
        neg 1
        neg 2
        pos 0 (bg)
        pos 1
        pos 2
        '''

        region_index = cond_index if cond_index < cond_nums//2 else cond_index - cond_nums//2
#        logger.info(f"{cond_index=}")
#        logger.info(f"{cond_nums=}")
#        logger.info(f"{region_index=}")


        for i,net in enumerate(self.networks):
            if region_index in net["region"]:
                scale = net["schedule"][frame_no]
                if scale > 0:
                    net["network"].active( scale )
#                    logger.info(f"{i=} active {scale=}")
                else:
                    net["network"].deactive( )
#                    logger.info(f"{i=} DEactive")

            else:
                net["network"].deactive( )
 #               logger.info(f"{i=} DEactive")

    def unapply(
            self,
        ):

        for net in self.networks:
            net["network"].deactive( )

