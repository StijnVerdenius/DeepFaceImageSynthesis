import os


def ensure_current_directory():
    """
    ensures we run from main directory even when we run testruns

    :return:
    """

    current_dir = os.getcwd()
    os.chdir(current_dir.split("DeepFakes")[0] + "DeepFakes/")
