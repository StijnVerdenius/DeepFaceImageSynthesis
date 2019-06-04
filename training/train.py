from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.statistic import Statistic
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
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

    pass  # todo: create


def combine_real_and_fake(real, fake, device):
    """
    Combines a set of real and fake images along the batch dimension
    Also generates targets.

    :param real:
    :param fake:
    :param device:
    :return:
    """


    # random indices
    shuffle_indices = list(range(int(real.shape[0] * 2)))
    random.shuffle(shuffle_indices)
    shuffle_indices = torch.LongTensor(shuffle_indices).to(device)

    # combine fake and real images
    composite = torch.cat((fake, real), dim=0).index_select(0, shuffle_indices)

    # combine real and fake targets
    labels = (torch.zeros(fake.shape[0]).to(device), torch.ones(real.shape[0]).to(device))
    ground_truth = torch.cat(labels, dim=0).index_select(0, shuffle_indices).to(device)

    return composite, ground_truth


def training_iteration(dataloader,
                       loss_function_gen: GeneralLoss,
                       loss_function_dis: GeneralLoss,
                       embedder: GeneralEmbedder,
                       generator: GeneralGenerator,
                       discriminator: GeneralDiscriminator,
                       arguments,
                       optimizer_dis,
                       optimizer_gen,
                       trainer_dis: Trainer,
                       trainer_gen: Trainer,
                       epoch_num):
    """
    one epoch

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

    for i, (batch, landmarks) in enumerate(dataloader):  # todo: how to split the data @ Klaus

        # prepare input
        batch.to(device)
        landmarks.to(device)
        combined_generator_input = torch.cat((batch, landmarks), dim=3) # concatenate in the channel-dimension?

        # set generator to train and discriminator to evaluation
        trainer_gen.prepare_training()
        trainer_dis.prepare_evaluation()

        # forward pass generator
        fake = generator.forward(combined_generator_input)
        loss_gen = loss_function_gen.forward(fake, discriminator)

        # backward pass generator
        trainer_gen.do_backward(loss_gen)

        # set generator to evaluation and discriminator to train
        trainer_gen.prepare_evaluation()
        trainer_dis.prepare_training()

        # combine real and fake images
        combined_set, labels = combine_real_and_fake(batch, fake, device)

        # forward pass discriminator
        predictions = discriminator.forward(combined_set)
        loss_dis = loss_function_dis.forward(predictions, labels)

        # backward discriminator
        trainer_dis.do_backward(loss_dis)

        # print progress to terminal
        batches_passed = i + (epoch_num * len(dataloader))
        if (batches_passed % arguments.eval_freq == 0):
            statistic = log(dataloader, loss_gen.item(), loss_dis.item(), embedder, generator, discriminator, arguments)
            progress.append(statistic)

        # save a set of pictures
        if (batches_passed % arguments.plot_freq == 0):
            plot_some_pictures(arguments.feedback)

    return progress


def log(dataloader, loss_gen, loss_dis, embedder, generator, discriminator, arguments) -> Statistic:
    """
    logs to terminal and calculate log_statistics

    # todo: validation-set?

    :param dataloader: validationset?
    :param loss:
    :param embedder:
    :param generator:
    :param discriminator:
    :param arguments:
    :return:
    """

    # print in-place with 5 decimals
    print(f"\r loss-generator: {loss_gen:0.5f}, loss-discriminator: {loss_dis:0.5f}", end='')

    # define training statistic
    return Statistic(loss_gen=loss_gen, loss_dis=loss_dis)


def train(dataloader,
          loss_gen: GeneralLoss,
          loss_dis: GeneralLoss,
          embedder: GeneralEmbedder,
          generator: GeneralGenerator,
          discriminator: GeneralDiscriminator,
          arguments,
          optimizer_gen,
          optimizer_dis):
    """
     main training function

    :param loss_gen:
    :param loss_dis:
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
            epoch_progress = training_iteration(dataloader,
                                            loss_gen,
                                            loss_dis,
                                            embedder,
                                            generator,
                                            discriminator,
                                            arguments,
                                            optimizer_dis,
                                            optimizer_gen,
                                            trainer_dis,
                                            trainer_gen,
                                            epoch)

            progress += epoch_progress

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
