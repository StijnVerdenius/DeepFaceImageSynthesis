from models.discriminators import *
from models.embedders import *
from models.generators import *
from models.losses import *
import importlib
import os
import sys
from utils.general_utils import ensure_current_directory
import inspect

types = ["discriminators", "embedders", "generators", "losses"]
models = {x: {} for x in types}


def read_all_classnames():
    for typ in types:
        for name in os.listdir(f"./models/{typ}"):
            if (not "__" in name):
                short_name = name.split(".")[0]
                module = importlib.import_module(f"models.{typ}.{short_name}")
                class_reference = getattr(module, short_name)
                models[typ][short_name] = class_reference


def find_right_model(type: str, name: str, **kwargs):
    """
    todo: returns model with arguments given a string name-tag

    :param type:
    :param name:
    :param kwargs:
    :return:
    """

    return models[type][name](**kwargs)

read_all_classnames()

if __name__ == '__main__':
    # unit-test functions here
    ensure_current_directory()
    read_all_classnames()
    z = find_right_model("losses", "GeneralLoss")
    print(type(z))
