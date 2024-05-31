import logging
from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from torch import Tensor

from animatediff import get_dir

EMBED_DIR = get_dir("data").joinpath("embeddings")
EMBED_DIR_SDXL = get_dir("data").joinpath("sdxl_embeddings")
EMBED_EXTS = [".pt", ".pth", ".bin", ".safetensors"]

logger = logging.getLogger(__name__)


def scan_text_embeddings(is_sdxl=False) -> list[Path]:
    embed_dir=EMBED_DIR_SDXL if is_sdxl else EMBED_DIR
    return [x for x in embed_dir.rglob("**/*") if x.is_file() and x.suffix.lower() in EMBED_EXTS]


def get_text_embeddings(return_tensors: bool = True, is_sdxl:bool = False) -> dict[str, Union[Tensor, Path]]:
    embed_dir=EMBED_DIR_SDXL if is_sdxl else EMBED_DIR
    embeds = {}
    skipped = {}
    path: Path
    for path in scan_text_embeddings(is_sdxl):
        if path.stem not in embeds:
            # new token/name, add it
            logger.debug(f"Found embedding token {path.stem} at {path.relative_to(embed_dir)}")
            embeds[path.stem] = path
        else:
            # duplicate token/name, skip it
            skipped[path.stem] = path
            logger.debug(f"Duplicate embedding token {path.stem} at {path.relative_to(embed_dir)}")

    # warn the user if there are duplicates we skipped
    if skipped:
        logger.warn(f"Skipped {len(skipped)} embeddings with duplicate tokens!")
        logger.warn(f"Skipped paths: {[x.relative_to(embed_dir) for x in skipped.values()]}")
        logger.warn("Rename these files to avoid collisions!")

    # we can optionally return the tensors instead of the paths
    if return_tensors:
        # load the embeddings
        embeds = {k: load_embed_weights(v) for k, v in embeds.items()}
        # filter out the ones that failed to load
        loaded_embeds = {k: v for k, v in embeds.items() if v is not None}
        if len(loaded_embeds) != len(embeds):
            logger.warn(f"Failed to load {len(embeds) - len(loaded_embeds)} embeddings!")
            logger.warn(f"Skipped embeddings: {[x for x in embeds.keys() if x not in loaded_embeds]}")

    # return a dict of {token: path | embedding}
    return embeds


def load_embed_weights(path: Path, key: Optional[str] = None) -> Optional[Tensor]:
    """Load an embedding from a file.
    Accepts an optional key to load a specific embedding from a file with multiple embeddings, otherwise
    it will try to load the first one it finds.
    """
    if not path.exists() and path.is_file():
        raise ValueError(f"Embedding path {path} does not exist or is not a file!")
    try:
        if path.suffix.lower() == ".safetensors":
            state_dict = load_file(path, device="cpu")
        elif path.suffix.lower() in EMBED_EXTS:
            state_dict = torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        logger.error(f"Failed to load embedding {path}", exc_info=True)
        return None

    embedding = None
    if len(state_dict) == 1:
        logger.debug(f"Found single key in {path.stem}, using it")
        embedding = next(iter(state_dict.values()))
    elif key is not None and key in state_dict:
        logger.debug(f"Using passed key {key} for {path.stem}")
        embedding = state_dict[key]
    elif "string_to_param" in state_dict:
        logger.debug(f"A1111 style embedding found for {path.stem}")
        embedding = next(iter(state_dict["string_to_param"].values()))
    else:
        # we couldn't find the embedding key, warn the user and just use the first key that's a Tensor
        logger.warn(f"Could not find embedding key in {path.stem}!")
        logger.warn("Taking a wild guess and using the first Tensor we find...")
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                embedding = value
                logger.warn(f"Using key: {key}")
                break

    return embedding


def load_text_embeddings(
    pipeline: DiffusionPipeline, text_embeds: Optional[tuple[str, torch.Tensor]] = None, is_sdxl = False
) -> None:
    if text_embeds is None:
        text_embeds = get_text_embeddings(False, is_sdxl)
    if len(text_embeds) < 1:
        logger.info("No TI embeddings found")
        return

    logger.info(f"Loading {len(text_embeds)} TI embeddings...")
    loaded, skipped, failed = [], [], []

    if True:
        vocab = pipeline.tokenizer.get_vocab()  # get the tokenizer vocab so we can skip loaded embeddings
        for token, emb_path in text_embeds.items():
            try:
                if token not in vocab:
                    if is_sdxl:
                        embed = load_embed_weights(emb_path, "clip_g").to(pipeline.text_encoder_2.device)
                        pipeline.load_textual_inversion(embed, token=token, text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
                        embed = load_embed_weights(emb_path, "clip_l").to(pipeline.text_encoder.device)
                        pipeline.load_textual_inversion(embed, token=token, text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
                    else:
                        embed = load_embed_weights(emb_path).to(pipeline.text_encoder.device)
                        pipeline.load_textual_inversion({token: embed})
                    logger.debug(f"Loaded embedding '{token}'")
                    loaded.append(token)
                else:
                    logger.debug(f"Skipping embedding '{token}' (already loaded)")
                    skipped.append(token)
            except Exception:
                logger.error(f"Failed to load TI embedding: {token}", exc_info=True)
                failed.append(token)

    else:
        vocab = pipeline.tokenizer.get_vocab()  # get the tokenizer vocab so we can skip loaded embeddings
        for token, embed in text_embeds.items():
            try:
                if token not in vocab:
                    if is_sdxl:
                        pipeline.load_textual_inversion(text_encoder_sd, token=token, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                    else:
                        pipeline.load_textual_inversion({token: embed})
                    logger.debug(f"Loaded embedding '{token}'")
                    loaded.append(token)
                else:
                    logger.debug(f"Skipping embedding '{token}' (already loaded)")
                    skipped.append(token)
            except Exception:
                logger.error(f"Failed to load TI embedding: {token}", exc_info=True)
                failed.append(token)

    # Print a summary of what we loaded
    logger.info(f"Loaded {len(loaded)} embeddings, {len(skipped)} existing, {len(failed)} failed")
    logger.info(f"Available embeddings: {', '.join(loaded + skipped)}")
    if len(failed) > 0:
        # only print failed if there were failures
        logger.warn(f"Failed to load embeddings: {', '.join(failed)}")
