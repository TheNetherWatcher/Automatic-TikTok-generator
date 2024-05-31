import logging
from functools import wraps
from pathlib import Path
from typing import Optional, TypeVar

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from torch import nn

from animatediff import HF_HUB_CACHE, HF_MODULE_REPO, get_dir
from animatediff.settings import CKPT_EXTENSIONS
from animatediff.utils.huggingface import get_hf_pipeline, get_hf_pipeline_sdxl
from animatediff.utils.util import path_from_cwd

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

# for the nop_train() monkeypatch
T = TypeVar("T", bound=nn.Module)


def nop_train(self: T, mode: bool = True) -> T:
    """No-op for monkeypatching train() call to prevent unfreezing module"""
    return self


def get_base_model(model_name_or_path: str, local_dir: Path, force: bool = False, is_sdxl:bool=False) -> Path:
    model_name_or_path = Path(model_name_or_path)

    model_save_dir = local_dir.joinpath(str(model_name_or_path).split("/")[-1]).resolve()
    model_is_repo_id = False if model_name_or_path.joinpath("model_index.json").exists() else True

    # if we have a HF repo ID, download it
    if model_is_repo_id:
        logger.debug("Base model is a HuggingFace repo ID")
        if model_save_dir.joinpath("model_index.json").exists():
            logger.debug(f"Base model already downloaded to: {path_from_cwd(model_save_dir)}")
        else:
            logger.info(f"Downloading base model from {model_name_or_path}...")
            if is_sdxl:
                _ = get_hf_pipeline_sdxl(model_name_or_path, model_save_dir, save=True, force_download=force)
            else:
                _ = get_hf_pipeline(model_name_or_path, model_save_dir, save=True, force_download=force)
        model_name_or_path = model_save_dir

    return Path(model_name_or_path)


def fix_checkpoint_if_needed(checkpoint: Path, debug:bool):
    def dump(loaded):
        for a in loaded:
            logger.info(f"{a} {loaded[a].shape}")

    if debug:
        from safetensors.torch import load_file, save_file
        loaded = load_file(checkpoint, "cpu")

        dump(loaded)

        return

    try:
        pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=str(checkpoint.absolute()),
            local_files_only=False,
            load_safety_checker=False,
        )
        logger.info("This file works fine.")
        return
    except:
        from safetensors.torch import load_file, save_file

        loaded = load_file(checkpoint, "cpu")

        convert_table_bias={
            "first_stage_model.decoder.mid.attn_1.to_k.bias":"first_stage_model.decoder.mid.attn_1.k.bias",
            "first_stage_model.decoder.mid.attn_1.to_out.0.bias":"first_stage_model.decoder.mid.attn_1.proj_out.bias",
            "first_stage_model.decoder.mid.attn_1.to_q.bias":"first_stage_model.decoder.mid.attn_1.q.bias",
            "first_stage_model.decoder.mid.attn_1.to_v.bias":"first_stage_model.decoder.mid.attn_1.v.bias",
            "first_stage_model.encoder.mid.attn_1.to_k.bias":"first_stage_model.encoder.mid.attn_1.k.bias",
            "first_stage_model.encoder.mid.attn_1.to_out.0.bias":"first_stage_model.encoder.mid.attn_1.proj_out.bias",
            "first_stage_model.encoder.mid.attn_1.to_q.bias":"first_stage_model.encoder.mid.attn_1.q.bias",
            "first_stage_model.encoder.mid.attn_1.to_v.bias":"first_stage_model.encoder.mid.attn_1.v.bias",
        }

        convert_table_weight={
            "first_stage_model.decoder.mid.attn_1.to_k.weight":"first_stage_model.decoder.mid.attn_1.k.weight",
            "first_stage_model.decoder.mid.attn_1.to_out.0.weight":"first_stage_model.decoder.mid.attn_1.proj_out.weight",
            "first_stage_model.decoder.mid.attn_1.to_q.weight":"first_stage_model.decoder.mid.attn_1.q.weight",
            "first_stage_model.decoder.mid.attn_1.to_v.weight":"first_stage_model.decoder.mid.attn_1.v.weight",
            "first_stage_model.encoder.mid.attn_1.to_k.weight":"first_stage_model.encoder.mid.attn_1.k.weight",
            "first_stage_model.encoder.mid.attn_1.to_out.0.weight":"first_stage_model.encoder.mid.attn_1.proj_out.weight",
            "first_stage_model.encoder.mid.attn_1.to_q.weight":"first_stage_model.encoder.mid.attn_1.q.weight",
            "first_stage_model.encoder.mid.attn_1.to_v.weight":"first_stage_model.encoder.mid.attn_1.v.weight",
        }

        for a in list(loaded.keys()):
            if a in convert_table_bias:
                new_key = convert_table_bias[a]
                loaded[new_key] = loaded.pop(a)
            elif a in convert_table_weight:
                new_key = convert_table_weight[a]
                item = loaded.pop(a)
                if len(item.shape) == 2:
                    item = item.unsqueeze(dim=-1).unsqueeze(dim=-1)
                loaded[new_key] = item

        new_path = str(checkpoint.parent / checkpoint.stem) + "_fixed"+checkpoint.suffix

        logger.info(f"Saving file to {new_path}")
        save_file(loaded, Path(new_path))



def checkpoint_to_pipeline(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionPipeline:
    logger.debug(f"Converting checkpoint {path_from_cwd(checkpoint)}")
    if target_dir is None:
        target_dir = pipeline_dir.joinpath(checkpoint.stem)

    pipeline = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=False,
        load_safety_checker=False,
    )

    if save:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
        pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline, target_dir

def checkpoint_to_pipeline_sdxl(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionXLPipeline:
    logger.debug(f"Converting checkpoint {path_from_cwd(checkpoint)}")
    if target_dir is None:
        target_dir = pipeline_dir.joinpath(checkpoint.stem)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=False,
        load_safety_checker=False,
    )

    if save:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
        pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline, target_dir

def get_checkpoint_weights(checkpoint: Path):
    temp_pipeline: StableDiffusionPipeline
    temp_pipeline, _ = checkpoint_to_pipeline(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, vae_state_dict

def get_checkpoint_weights_sdxl(checkpoint: Path):
    temp_pipeline: StableDiffusionXLPipeline
    temp_pipeline, _ = checkpoint_to_pipeline_sdxl(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    tenc2_state_dict = temp_pipeline.text_encoder_2.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, tenc2_state_dict, vae_state_dict


def ensure_motion_modules(
    repo_id: str = HF_MODULE_REPO,
    fp16: bool = False,
    force: bool = False,
):
    """Retrieve the motion modules from HuggingFace Hub."""
    module_files = ["mm_sd_v14.safetensors", "mm_sd_v15.safetensors"]
    module_dir = get_dir("data/models/motion-module")
    for file in module_files:
        target_path = module_dir.joinpath(file)
        if fp16:
            target_path = target_path.with_suffix(".fp16.safetensors")
        if target_path.exists() and force is not True:
            logger.debug(f"File {path_from_cwd(target_path)} already exists, skipping download")
        else:
            result = hf_hub_download(
                repo_id=repo_id,
                filename=target_path.name,
                cache_dir=HF_HUB_CACHE,
                local_dir=module_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.debug(f"Downloaded {path_from_cwd(result)}")
