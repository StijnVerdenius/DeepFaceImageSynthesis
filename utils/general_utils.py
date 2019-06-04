import os

from utils.constants import *
import torch


def ensure_current_directory():
    """
    ensures we run from main directory even when we run testruns

    :return:
    """

    current_dir = os.getcwd()
    os.chdir(current_dir.split("DeepFakes")[0] + "DeepFakes/")


def setup_directories():
    stamp = DATA_MANAGER.date_stamp()
    dirs = [PIC_DIR, PROGRESS_DIR, MODELS_DIR]
    for dir_to_be in dirs:
        DATA_MANAGER.create_dir(f"output/{stamp}/{dir_to_be}")


def get_device(requested_device):
    # get right device
    device = torch.device("cpu")
    if ("cuda" in requested_device):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
