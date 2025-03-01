import os
import shutil
import re
import json
import yaml
from huggingface_hub import Repository, create_repo
from pathlib import Path
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

def freeze_model(model, partname='classifier'):
    for name,p in model.named_parameters():
        if not name.startswith(partname):
            p.requires_grad = False

def calculate_params(model):
    num_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(f"{num_params = :,} | {trainable_params = :,}")
    
def load_config(config_path):
    """Loads configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None

def load_config_relativefolder(filename="config.yaml"):
    """
    Loads configuration parameters from a YAML file.

    Args:
        filename (str): The name or path of the YAML file to load.

    Returns:
        dict: A dictionary containing the configuration parameters, or None if an error occurs.
    """
    try:
        # Check if the filename is a path
        if os.path.isabs(filename) or os.path.exists(filename):
            config_path = filename
        else:
            # Get the directory of the current script and combine with the filename
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if not filename.lower().endswith(".yaml"):
                filename += ".yaml"
            config_path = os.path.join(script_dir, "configs", filename)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def pushtohub(hub_model_id, output_dir, hub_token):
    # Retrieve of infer repo_name
    repo_name = hub_model_id
    if repo_name is None:
        repo_name = Path(output_dir).absolute().name
    # Create repo and retrieve repo_id
    repo_id = create_repo(repo_name, exist_ok=True, token=hub_token).repo_id
    # Clone repo locally
    repo = Repository(output_dir, clone_from=repo_id, token=hub_token)

    with open(os.path.join(output_dir, ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
            gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
            gitignore.write("epoch_*\n")