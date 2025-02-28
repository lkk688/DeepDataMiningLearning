import os
import shutil
import re
import json
import logging
#creates a logger for the current module
logger = logging.getLogger(__name__)

#from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def load_hfcheckpoint(checkpoint_dir, overwrite_output_dir=False):
    last_checkpoint = None
    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir) and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is None and len(os.listdir(checkpoint_dir)) > 0:
            raise ValueError(
                f"Output directory ({checkpoint_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint