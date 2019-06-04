from models.discriminators import *
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders import *
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.data_management import DataManager
from models.generators import *
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses import *
from training.train import MODELS_DIR
import importlib
import os
import sys
from utils.general_utils import ensure_current_directory
import inspect

types = ["discriminators", "embedders", "generators", "losses"]
models = {x: {} for x in types}


def _read_all_classnames():
    """
    private function that imports all class references in a dictionary

    :return:
    """

    for typ in types:
        for name in os.listdir(f"./models/{typ}"):
            if (not "__" in name):
                short_name = name.split(".")[0]
                module = importlib.import_module(f"models.{typ}.{short_name}")
                class_reference = getattr(module, short_name)
                models[typ][short_name] = class_reference


def find_right_model(type: str, name: str, **kwargs):
    """
    returns model with arguments given a string name-tag

    :param type:
    :param name:
    :param kwargs:
    :return:
    """

    return models[type][name](**kwargs)


def save_models(discriminator: GeneralDiscriminator, generator: GeneralGenerator, embedder: GeneralEmbedder,
                suffix: str, data_manager: DataManager):
    """
    Saves current state of models

    :param discriminator:
    :param generator:
    :param embedder:
    :param suffix: determines file name
    :param data_manager: from training (contains the right date_stamp_directory)
    :return:
    """
    save_dict = {"discriminator": discriminator.state_dict(), "generator": generator.state_dict(),
                 "embedder": embedder.state_dict()}

    data_manager.save_python_obj(save_dict, f"results/output/{data_manager.stamp}/{MODELS_DIR}/{suffix}")


def load_models_and_state(discriminator: GeneralDiscriminator, generator: GeneralGenerator, embedder: GeneralEmbedder,
                          suffix: str, stamp: str):
    """
    Loads saved models given a suffix and then also loads in state dicts already

    :param discriminator: fully initialized
    :param generator: fully initialized
    :param embedder: fully initialized
    :param suffix: filename
    :param stamp: date_Stamp_directory
    :return:
    """

    data_manager = DataManager("./results/")

    models = data_manager.load_python_obj(f"results/output/{stamp}/{MODELS_DIR}/{suffix}")

    discriminator.load_state_dict(models["discriminator"])
    embedder.load_state_dict(models["embedder"])
    generator.load_state_dict(models["generator"])

    return discriminator, generator, embedder


def load_states(suffix: str, stamp: str):
    """
    Only loads state dicts

    :param suffix: filename
    :param stamp: date_stamp
    :return:
    """

    data_manager = DataManager("./results/")
    return data_manager.load_python_obj(f"results/output/{stamp}/{MODELS_DIR}/{suffix}")


# needed to load in class references
_read_all_classnames()

if __name__ == '__main__':
    # unit-test functions here
    ensure_current_directory()
    _read_all_classnames()
    z = find_right_model("losses", "GeneralLoss")
    print(type(z))
