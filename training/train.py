from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.statistic import Statistic
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from utils.general_utils import ensure_current_directory, setup_directories, mean
from utils.constants import *
from models.general.trainer import Trainer
from utils.model_utils import save_models
import random
import torch
from torchvision.utils import save_image


def plot_some_pictures(feedback, images, batches_done):
    """
    save some plots in PIC_DIR

    """

    save_image(images[:25].view(-1, 3, IMSIZE, IMSIZE),
               f'results/output/{DATA_MANAGER.stamp}/{PIC_DIR}/{batches_done}.png',
               nrow=5, normalize=True)

    if (feedback):
        # TODO: if feedback is on, run the following script from the result-image directory in terminal while it is running:
        # watch xdg-open latests_plot.png
        save_image(images[:25].view(-1, 3, IMSIZE, IMSIZE),
                   f'results/output/{DATA_MANAGER.stamp}/{PIC_DIR}/latests_plot.png',
                   nrow=5, normalize=True)


def combine_real_and_fake(real, fake):
    """
    Combines a set of real and fake images along the batch dimension
    Also generates targets.

    """

    # random indices
    shuffle_indices = list(range(int(real.shape[0] * 2)))
    random.shuffle(shuffle_indices)
    shuffle_indices = torch.LongTensor(shuffle_indices).to(DEVICE)

    # combine fake and real images
    composite = torch.cat((fake, real), dim=0).index_select(0, shuffle_indices)

    # combine real and fake targets
    labels = (torch.zeros(fake.shape[0]).to(DEVICE), torch.ones(real.shape[0]).to(DEVICE))
    ground_truth = torch.cat(labels, dim=0).index_select(0, shuffle_indices).to(DEVICE)

    return composite, ground_truth


def batch_iteration(batch,
                    landmarks,
                    discriminator,
                    generator,
                    loss_function_gen,
                    loss_function_dis,
                    trainer_dis,
                    trainer_gen,
                    train=True):
    """
     inner loop of epoch iteration

    """

    # prepare input
    batch.to(DEVICE)
    landmarks.to(DEVICE)
    landmarked_batch = torch.cat((batch, landmarks), dim=3)  # concatenate in the channel-dimension?

    if (train):
        # set generator to train and discriminator to evaluation
        trainer_gen.prepare_training()
        trainer_dis.prepare_evaluation()

    # forward pass generator
    fake = generator.forward(landmarked_batch)
    loss_gen = loss_function_gen.forward(fake, discriminator)

    if (train):
        # backward pass generator
        trainer_gen.do_backward(loss_gen)

        # set discriminator to train
        trainer_dis.prepare_training()

    # combine real and fake
    landmarked_fake = torch.cat((fake, landmarks), dim=3)  # concatenate in the channel-dimension?
    combined_set, labels = combine_real_and_fake(landmarked_batch, landmarked_fake)

    # forward pass discriminator
    predictions = discriminator.forward(combined_set)
    loss_dis = loss_function_dis.forward(predictions, labels)

    if (train):
        # backward discriminator
        trainer_dis.do_backward(loss_dis)

    return loss_gen.item(), loss_dis.item(), fake, predictions, labels


def epoch_iteration(dataloader_train,
                    dataloader_validate,
                    loss_function_gen: GeneralLoss,
                    loss_function_dis: GeneralLoss,
                    embedder: GeneralEmbedder,
                    generator: GeneralGenerator,
                    discriminator: GeneralDiscriminator,
                    arguments,
                    trainer_dis: Trainer,
                    trainer_gen: Trainer,
                    epoch_num
                    ):
    """
    one epoch implementation

    """

    progress = []

    for i, (batch, landmarks) in enumerate(dataloader_train):  # todo: how to split the data @ Klaus

        # run batch iteration
        loss_gen, loss_dis, fake_images, _, _ = batch_iteration(batch,
                                                                landmarks,
                                                                discriminator,
                                                                generator,
                                                                loss_function_gen,
                                                                loss_function_dis,
                                                                trainer_dis,
                                                                trainer_gen
                                                                )

        # calculate amount of passed batches
        batches_passed = i + (epoch_num * len(dataloader_train))

        # print progress to terminal
        if (batches_passed % arguments.eval_freq == 0):
            # log to terminal and retrieve a statistics object
            statistic = log(dataloader_validate,
                            loss_gen,
                            loss_dis,
                            embedder,
                            generator,
                            discriminator,
                            loss_function_gen,
                            loss_function_dis,
                            trainer_dis,
                            trainer_gen
                            )

            # append statistic to list
            progress.append(statistic)

        # save a set of pictures
        if (batches_passed % arguments.plot_freq == 0):
            plot_some_pictures(arguments.feedback, fake_images, batches_passed)

    return progress


def calculate_accuracy(predictions, targets):
    """
    Gets the accuracy for discriminator

    """

    actual_predictions = predictions > 0.5
    true_positives = (actual_predictions == (targets > 0.5)).type(torch.DoubleTensor)
    accuracy = (torch.mean(true_positives))

    return accuracy.item()


def validate(validation_set, embedder, generator, discriminator, loss_function_dis, loss_function_gen):
    """
    Runs a validation epoch

    """

    # init
    total_loss_discriminator = []
    total_loss_generator = []
    total_accuracy = []

    for i, (batch, landmarks) in validation_set:
        # run batch iteration
        loss_gen, loss_dis, _, predictions, actual_labels = batch_iteration(batch,
                                                                            landmarks,
                                                                            discriminator,
                                                                            generator,
                                                                            loss_function_gen,
                                                                            loss_function_dis,
                                                                            None,
                                                                            None,
                                                                            train=False
                                                                            )

        # also get accuracy
        accuracy = calculate_accuracy(predictions, actual_labels)

        # append findings to respective lists
        total_accuracy.append(accuracy)
        total_loss_discriminator.append(loss_dis)
        total_loss_discriminator.append(loss_gen)

    return mean(total_loss_generator), mean(total_loss_discriminator), mean(total_accuracy)


def log(validation_set,
        loss_gen: int,
        loss_dis: int,
        embedder: GeneralEmbedder,
        generator: GeneralGenerator,
        discriminator: GeneralDiscriminator,
        loss_function_gen: GeneralLoss,
        loss_function_dis: GeneralLoss,
        trainer_dis: Trainer,
        trainer_gen: Trainer
        ) -> Statistic:
    """
    logs to terminal and calculate log_statistics

    """

    # put models in evaluation mode
    trainer_dis.prepare_evaluation()
    trainer_gen.prepare_evaluation()

    # validate on validationset
    loss_gen_validate, loss_dis_validate, discriminator_accuracy = validate(validation_set,
                                                                            embedder,
                                                                            generator,
                                                                            discriminator,
                                                                            loss_function_dis,
                                                                            loss_function_gen,
                                                                            )

    # print in-place with 3 decimals
    print(
        f"\r",
        f"loss-generator-train: {loss_gen:0.3f}",
        f"loss-discriminator-train: {loss_dis:0.3f}",
        f"loss-generator-validate: {loss_gen_validate:0.3f}",
        f"loss-discriminator-validate: {loss_dis_validate:0.3f}",
        f"accuracy-discriminator: {discriminator_accuracy}",
        end='')

    # define training statistic
    return Statistic(loss_gen_train=loss_gen,
                     loss_dis_train=loss_dis,
                     loss_gen_val=loss_gen_validate,
                     loss_dis_val=loss_dis_validate,
                     dis_acc=discriminator_accuracy)


def train(dataloader_train,
          dataloader_validate,
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
        trainer_gen = Trainer([generator], [optimizer_gen])
        trainer_dis = Trainer([discriminator], [optimizer_dis])

        # run
        for epoch in range(arguments.epochs):
            epoch_progress = epoch_iteration(dataloader_train,
                                             dataloader_validate,
                                             loss_gen,
                                             loss_dis,
                                             embedder,
                                             generator,
                                             discriminator,
                                             arguments,
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
