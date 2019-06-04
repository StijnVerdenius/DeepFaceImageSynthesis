from models.general.statistic import Statistic
from utils.general_utils import ensure_current_directory, setup_directories, get_device
from utils.constants import *
from models.general.trainer import Trainer
from training.finetune import *
from training.meta_train import *
from utils.model_utils import save_models
import random
import torch


# todo: in this file put functions that are shared for both fine tuning and meta training

def plot_some_pictures(feedback):
    """
    save some plots in PIC_DIR
    :return:
    """

    date_directory = DATA_MANAGER.stamp

    pass  # todo: do something


def training_iteration(dataloader, loss_function, embedder, generator, discriminator, arguments, optimizer_dis,
                       optimizer_gen,
                       trainer_dis, trainer_gen, epoch_num):
    """
    one epoch (template) TODO: this function obviously needs work, but this really depends on the goal we eventually set,
     todo:  so please see it just as an example solution and not as a proposed solution

    :param dataloader:
    :param loss_function:
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

    device = get_device(arguments.device)

    progress = []

    for i, (batch, _, _) in enumerate(dataloader):  # todo: how to split the data

        ground_truth_landmarks = None  # todo

        # preset everything for training
        trainer_gen.prepare_training()
        trainer_dis.prepare_training()

        # generator images todo: i know we have to work on this still, its just a suggestion
        embedded = embedder.forward(batch)
        fake = generator.forward(embedded, ground_truth_landmarks)

        # random indices
        shuffle_indices = list(range(int(batch.shape[0] * 2)))
        random.shuffle(shuffle_indices)
        shuffle_indices = torch.LongTensor(shuffle_indices).to(device)

        # combine fake and real images and also their targets in a composite input for discriminator
        composite = torch.cat((fake, batch), dim=0).index_select(0, shuffle_indices)
        labels = (torch.zeros(fake.shape[0]).to(device), torch.ones(batch.shape[0]).to(device))
        ground_truth = torch.cat(labels, dim=0).index_select(0, shuffle_indices).to(device)

        # discriminator forward pass todo: i know we have to work on this still, its just a suggestion
        discriminator_output = discriminator.forward(composite)


        # combined loss function, todo: i know we have to work on this still, its just a suggestion
        loss = loss_function.forward(discriminator_output, ground_truth, embedded,
                                     fake)  # i dunno whats gonna go in here but just see it as anything can go in

        # backward passes todo: i dont know if we can actually call backward on the same loss twice? depends a bit on implementation as well
        trainer_gen.do_backward(loss)
        trainer_dis.do_backward(loss)

        # print progress to terminal
        if (i + (epoch_num * len(dataloader)) % arguments.eval_freq == 0):
            statistic = log(dataloader, loss_function, embedder, generator, discriminator, arguments)
            progress.append(statistic)
            plot_some_pictures(arguments.feedback)

    return progress


def log(dataloader, loss, embedder, generator, discriminator, arguments) -> Statistic:
    """
    logs to terminal and calculate log_statistics # todo

    :param dataloader: validationset?
    :param loss:
    :param embedder:
    :param generator:
    :param discriminator:
    :param arguments:
    :return:
    """
    loss = 1
    print(f"\r loss: {loss:0.5f}", end='')  # print in-place with 5 decimals example

    return Statistic(loss=loss)


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

    # setup data output directories:
    setup_directories()

    # data gathering
    progress = []

    try:
        # setup
        trainer_gen = Trainer([embedder, generator], [optimizer_gen])
        trainer_dis = Trainer([discriminator], optimizer_gen)

        # run
        for epoch in range(arguments.epochs):
            epoch_loss = training_iteration(dataloader, loss, embedder, generator, discriminator, arguments,
                                            optimizer_dis,
                                            optimizer_gen, trainer_dis, trainer_gen, epoch)

            progress.append(epoch_loss)

            # write progress to pickle file (overwrite because there is no point keeping seperate versions)
            DATA_MANAGER.save_python_obj(progress, f"output/{DATA_MANAGER.stamp}/{PROGRESS_DIR}/progress_list")


    except KeyboardInterrupt:
        print("Killed by user")
        save_models(discriminator, generator, embedder, f"KILLED_at_epoch_{epoch}")
        return False
    except Exception as e:
        print(e)
        save_models(discriminator, generator, embedder, f"CRASH_at_epoch_{epoch}")
        raise e

    # example last save
    save_models(discriminator, generator, embedder, "finished")
    return True


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
