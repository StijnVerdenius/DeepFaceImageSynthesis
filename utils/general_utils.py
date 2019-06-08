import inspect
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
    dirs = OUTPUT_DIRS
    for dir_to_be in dirs:
        DATA_MANAGER.create_dir(f"{stamp}/{dir_to_be}")


def save_codebase_of_run(arguments):
    directory = f"./{PREFIX_OUTPUT}/{DATA_MANAGER.stamp}/{CODE_DIR}"
    f = open(f"{directory}/arguments.txt", "w")
    f.write(str(arguments).replace(", ", "\n"))
    f.close()

    stack = ["."]

    while len(stack) > 0:

        path = stack.pop(0)

        for file_name in os.listdir(os.path.join(os.getcwd(), path)):

            if file_name.endswith(".py"):
                f = open(f"{directory}/{file_name}".replace(".py", ""), "w")
                lines = open(f"{path}/{file_name}", "r").read()
                f.write(str(lines))
                f.close()
            elif (os.path.isdir(os.path.join(os.getcwd(), path, file_name))):
                stack.append(os.path.join(path, file_name))

    base = os.path.join(os.getcwd(), PREFIX_OUTPUT, DATA_MANAGER.stamp, CODE_DIR)
    for file_name in list(os.listdir(base)):
        if ("arguments.txt" in file_name): continue
        os.rename(base + "/" + file_name, base + "/" + file_name + ".py")


def assert_type(expectedType, content):
    """ makes sure type is respected"""

    func = inspect.stack()[1][3]
    assert isinstance(content, expectedType), "No {} entered in {} but instead value {}".format(str(expectedType), func,
                                                                                                str(content))


def assert_non_empty(content):
    """ makes sure not None or len()==0 """

    func = inspect.stack()[1][3]
    assert not content == None, "Content is null in {}".format(func)
    if (type(content) is list or type(content) == str):
        assert len(content) > 0, "Empty {} in {}".format(type(content), func)


def mean(input_list):
    assert_type(list, input_list)
    return sum(input_list) / len(input_list)
