import os

from utils.constants import *


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
        DATA_MANAGER.create_dir(f"{stamp}/{dir_to_be}")


def mean(input_list):
    return sum(input_list) / len(input_list)
