import glob
import json
import logging
import os
import re
import shutil
from pathlib import Path

from animatediff import get_dir

logger = logging.getLogger(__name__)

data_dir = get_dir("data")

extra_loading_regex = r'(<[^>]+?>)'

def generate_config_from_civitai_info(
    lora_dir:Path,
    config_org:Path,
    out_dir:Path,
    lora_weight:float,
):
    lora_abs_dir = lora_dir.absolute()
    config_org = config_org.absolute()
    out_dir = out_dir.absolute()

    civitais = sorted(glob.glob( os.path.join(lora_abs_dir, "*.civitai.info"), recursive=False))

    with open(config_org, "r") as cf:
        org_config = json.load(cf)

        for civ in civitais:

            logger.info(f"convert {civ}")

            with open(civ, "r") as f:
                # trim .civitai.info
                name = os.path.splitext(os.path.splitext(os.path.basename(civ))[0])[0]

                output_path = out_dir.joinpath(name + ".json")

                if os.path.isfile(output_path):
                    logger.info("already converted -> skip")
                    continue

                if os.path.isfile( lora_abs_dir.joinpath(name + ".safetensors")):
                    lora_path = os.path.relpath(lora_abs_dir.joinpath(name + ".safetensors"), data_dir)
                elif os.path.isfile( lora_abs_dir.joinpath(name + ".ckpt")):
                    lora_path = os.path.relpath(lora_abs_dir.joinpath(name + ".ckpt"), data_dir)
                else:
                    logger.info("lora file not found -> skip")
                    continue

                info = json.load(f)

                if not info:
                    logger.info(f"empty civitai info -> skip")
                    continue

                if info["model"]["type"] not in ("LORA","lora"):
                    logger.info(f"unsupported type {info['model']['type']} -> skip")
                    continue

                new_config = org_config.copy()

                new_config["name"] = name

                new_prompt_map = {}
                new_n_prompt = ""
                new_seed = -1


                raw_prompt_map = {}

                i = 0
                for img_info in info["images"]:
                    if img_info["meta"]:
                        try:
                            raw_prompt = img_info["meta"]["prompt"]
                        except Exception as e:
                            logger.info("missing prompt")
                            continue

                        raw_prompt_map[str(10000 + i*32)] = raw_prompt

                        new_prompt_map[str(i*32)] = re.sub(extra_loading_regex, '', raw_prompt)

                        if not new_n_prompt:
                            try:
                                new_n_prompt = img_info["meta"]["negativePrompt"]
                            except Exception as e:
                                new_n_prompt = ""
                        if new_seed == -1:
                            try:
                                new_seed = img_info["meta"]["seed"]
                            except Exception as e:
                                new_seed = -1

                        i += 1

                if not new_prompt_map:
                    new_prompt_map[str(0)] = ""

                for k in raw_prompt_map:
                    # comment
                    new_prompt_map[k] = raw_prompt_map[k]

                new_config["prompt_map"] = new_prompt_map
                new_config["n_prompt"] = [new_n_prompt]
                new_config["seed"] = [new_seed]

                new_config["lora_map"] = {lora_path.replace(os.sep,'/'):lora_weight}

                with open( out_dir.joinpath(name + ".json"), 'w') as wf:
                    json.dump(new_config, wf, indent=4)
                    logger.info("converted!")

                preview = lora_abs_dir.joinpath(name + ".preview.png")
                if preview.is_file():
                    shutil.copy(preview, out_dir.joinpath(name + ".preview.png"))


