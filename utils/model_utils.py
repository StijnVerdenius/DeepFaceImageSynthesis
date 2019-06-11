from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.generators.GeneralGenerator import GeneralGenerator
from utils.constants import *
import importlib
import os
from utils.general_utils import ensure_current_directory

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
                suffix: str):
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

    DATA_MANAGER.save_python_obj(save_dict, f"{DATA_MANAGER.stamp}/{MODELS_DIR}/{suffix}")


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

    models = DATA_MANAGER.load_python_obj(f"{stamp}/{MODELS_DIR}/{suffix}")

    discriminator.load_state_dict(models["discriminator"])
    # embedder.load_state_dict(models["embedder"])
    generator.load_state_dict(models["generator"])

    discriminator.to(DEVICE)
    embedder.to(DEVICE)
    generator.to(DEVICE)

    return discriminator, generator, embedder


def load_states(suffix: str, stamp: str):
    """
    Only loads state dicts

    :param suffix: filename
    :param stamp: date_stamp
    :return:
    """

    return DATA_MANAGER.load_python_obj(f"{stamp}/{MODELS_DIR}/{suffix}")


# needed to load in class references
_read_all_classnames()

if __name__ == '__main__':
    # unit-test functions here
    ensure_current_directory()
    _read_all_classnames()
    z = find_right_model("losses", "GeneralLoss")
    print(type(z))
