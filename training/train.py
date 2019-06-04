from utils.general_utils import ensure_current_directory
from models.general.trainer import Trainer
from models.general.data_management import DataManager
from training.finetune import *
from training.meta_train import *

PIC_DIR = "output_pictures"
MODELS_DIR = "saved_models"
PROGRESS_DIR = "training_progress"


# todo: in this file put functions that are shared for both fine tuning and meta training

def training_iteration(dataloader, loss, embedder, generator, discriminator, arguments, optimizer_dis, optimizer_gen,
                       trainer_dis, trainer_gen):
    """
    one epoch

    :param dataloader:
    :param loss:
    :param embedder:
    :param generator:
    :param discriminator:
    :param arguments:
    :param optimizer_dis:
    :param optimizer_gen:
    :param trainer_dis:
    :param trainer_gen:
    :return:
    """

    losses = []

    for i, (batch, y) in enumerate(dataloader):
        trainer_gen.prepare_training()

        pass  # todo

    return losses


def log(dataloader, loss, embedder, generator, discriminator, arguments):
    """
    logs to terminal

    :param dataloader:
    :param loss:
    :param embedder:
    :param generator:
    :param discriminator:
    :param arguments:
    :return:
    """
    i = 1
    print(f"\r some text {i}", end='')  # log in place example
    pass  # todo


def train(dataloader, loss, embedder, generator, discriminator, arguments, optimizer_gen, optimizer_dis):
    """
     main training function

    :param dataloader:
    :param loss:
    :param embedder:
    :param generator:
    :param discriminator:
    :param arguments:
    :param optimizer_gen:
    :param optimizer_dis:
    :return:
    """

    # set data manager:
    data_manager = DataManager("./results/")
    setup_directories(data_manager, data_manager.date_stamp())

    # data gathering
    losses = []

    try:
        # setup
        trainer_gen = Trainer([embedder, generator], [optimizer_gen])
        trainer_dis = Trainer([discriminator], optimizer_gen)

        # run
        for epoch in range(arguments.epochs):
            epoch_loss = training_iteration(dataloader, loss, embedder, generator, discriminator, arguments,
                                            optimizer_dis,
                                            optimizer_gen, trainer_dis, trainer_gen)

            losses.append(epoch_loss)


    except KeyboardInterrupt:
        # todo: handle force kill by user (lasts saves etc.)
        pass
    except Exception as e:
        print(e)
        # todo: handle unexpected failure
        pass

    # example last save
    data_manager.save_python_obj(losses, f"output/{data_manager.stamp}/{PROGRESS_DIR}/final_losses_list")


def setup_directories(data_manager: DataManager, stamp: str):
    dirs = [PIC_DIR, PROGRESS_DIR, MODELS_DIR]
    for dir_to_be in dirs:
        data_manager.create_dir(f"output/{stamp}/{dir_to_be}")


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
