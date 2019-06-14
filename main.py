# torch debug
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.data import DataLoader

from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from utils.general_utils import *
from utils.model_utils import find_right_model, load_models_and_state
from utils.constants import *
import argparse
from training.train import TrainingProcess
from testing.test import compare
import torch.optim as opt
import torch
from data.Dataset300VW import X300VWDataset
import numpy as np
import sys

torch.backends.cudnn.benchmark = True


def dummy_batch(batch_size, channels):
    return np.random.normal(0, 1, (batch_size, channels, IMSIZE, IMSIZE))


def load_data(keyword: str, batch_size: int, mode: str) -> DataLoader:  # todo @ klaus

    data = None

    if (keyword == "train"):
        data = DataLoader(X300VWDataset(), shuffle=(False or mode == "test"), batch_size=batch_size,
                          drop_last=True)  # Changed to false!!!

    elif (keyword == "validate"):
        data = DataLoader(X300VWDataset(), shuffle=(False or mode == "test"), batch_size=batch_size,
                          drop_last=True)  # Changed to false!!!
    elif (keyword == "debug"):
        data = [(dummy_batch(batch_size, INPUT_CHANNELS), dummy_batch(batch_size, INPUT_LANDMARK_CHANNELS)) for _ in
                range(5)]
    else:
        raise Exception(f"{keyword} is not a valid dataset")

    print(f"finished loading {keyword} of length: {len(data)}")

    return data


def main(arguments):
    # to measure the time needed
    pr = None
    if (arguments.timing):
        pr = start_timing()

    print(f"Device used = {DEVICE}")

    # data
    dataloader_train = load_data("train", arguments.batch_size, arguments.mode)
    dataloader_validate = load_data("validate", arguments.batch_size, arguments.mode)

    # get models
    embedder = find_right_model(EMBED_DIR, arguments.embedder,
                                device=DEVICE,
                                n_channels_in=INPUT_SIZE,
                                n_channels_out=arguments.embedding_size,
                                use_dropout=arguments.dropout).to(DEVICE)

    generator = find_right_model(GEN_DIR, arguments.generator,
                                 device=DEVICE,
                                 n_channels_in=INPUT_SIZE,
                                 use_dropout=arguments.dropout).to(DEVICE)

    discriminator = find_right_model(DIS_DIR, arguments.discriminator,
                                     device=DEVICE,
                                     n_channels_in=INPUT_SIZE,
                                     use_dropout=arguments.dropout).to(DEVICE)

    # assertions
    assert_type(GeneralGenerator, generator)
    assert_type(GeneralDiscriminator, discriminator)
    assert_type(GeneralEmbedder, embedder)

    # train or test
    if (arguments.mode == "train" or arguments.mode == "finetune"):

        # init optimizers
        generator_optimizer = find_right_model(OPTIMS, arguments.generator_optimizer, params=generator.parameters(),
                                               lr=arguments.learning_rate)
        discriminator_optimizer = find_right_model(OPTIMS, arguments.discriminator_optimizer,
                                                   params=discriminator.parameters(), lr=arguments.learning_rate)
        embedder_optimizer = find_right_model(OPTIMS, arguments.embedder_optimizer, params=embedder.parameters(),
                                              lr=arguments.learning_rate)

        # define loss functions
        if (not arguments.loss_gen == TOTAL_LOSS):
            print(
                f"{PRINTCOLOR_RED} WARNING: running with one generator-loss only: {arguments.loss_gen} {PRINTCOLOR_END}")
        weights_loss_functions = get_generator_loss_weights(arguments)
        loss_gen = find_right_model(LOSS_DIR, TOTAL_LOSS, **weights_loss_functions)
        loss_dis = find_right_model(LOSS_DIR, arguments.loss_dis)

        # assertions
        assert_type(GeneralLoss, loss_dis)
        assert_type(GeneralLoss, loss_gen)

        # define process
        train_progress = TrainingProcess(generator,
                                         discriminator,
                                         embedder,
                                         dataloader_train,
                                         dataloader_validate,
                                         generator_optimizer,
                                         discriminator_optimizer,
                                         embedder_optimizer,
                                         loss_gen,
                                         loss_dis,
                                         arguments)

        # train
        trained_succesfully = train_progress.train()

        # handle failure
        if (not trained_succesfully):
            pass  # todo

    elif (arguments.mode == "test"):

        # load in state dicts
        load_models_and_state(discriminator,
                              generator,
                              embedder,
                              arguments.test_model_suffix,
                              arguments.test_model_date)

        # run test
        compare(dataloader_validate, embedder, generator, arguments, number_of_batches=10, number_of_pictures=3)

    else:

        raise Exception(f"Unrecognized train/test mode?: {arguments.mode}")

    if (arguments.timing):
        stop_timing(pr)


def parse():
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--eval_freq', type=int, default=10, help='Frequency (batch-wise) of evaluation')
    parser.add_argument('--plot_freq', type=int, default=100, help='Frequency (batch-wise) of plotting pictures')
    parser.add_argument('--saving_freq', type=int, default=10, help='Frequency (epoch-wise) of saving models')
    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--feedback', default=False, type=bool, help='whether to plot or not during training')
    parser.add_argument('--mode', default="train", type=str, help="'train', 'test' or 'finetune'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=bool, default=False, help='Learning rate')
    parser.add_argument('--max_training_minutes', type=int, default=-1, help='After which process is killed automatically')

    # debug
    parser.add_argument('--timing', type=bool, default=False, help='are we measuring efficiency?')

    # test arguments
    parser.add_argument('--test_model_date', default="week_2", type=str,
                        help='date_stamp string for which model to load')
    parser.add_argument('--test_model_suffix', default="Models_at_epoch_49", type=str,
                        help='filename string for which model to load')

    # model arguments
    parser.add_argument('--embedding_size', default=2, type=int, help='dimensionality of latent embedding space')
    parser.add_argument('--embedder', default="EmptyEmbedder", type=str, help="name of objectclass")
    parser.add_argument('--discriminator', default="PatchDiscriminator", type=str, help="name of objectclass")
    parser.add_argument('--generator', default="ResnetGenerator", type=str, help="name of objectclass")

    # optimizer arguments
    parser.add_argument('--discriminator_optimizer', default="SGD", type=str, help="name of objectclass")
    parser.add_argument('--generator_optimizer', default="Adam", type=str, help="name of objectclass")
    parser.add_argument('--embedder_optimizer', default="Adam", type=str, help="name of objectclass")

    # loss arguments
    parser.add_argument('--loss_gen', default=TOTAL_LOSS, type=str,
                        help="Overwrites hyperparams generatorloss if not total")
    parser.add_argument('--loss_dis', default="HingeAdverserialDLoss", type=str, help="name of objectclass")

    # hyperparams generatorloss  (-1 === DEFAULT)
    parser.add_argument('--NonSaturatingGLoss_weight', default=10, type=float,
                        help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--PerceptualLoss_weight', default=-1, type=float,
                        help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--PixelLoss_weight', default=-1, type=float,
                        help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--ConsistencyLoss_weight', default=-1, type=float,
                        help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--TripleConsistencyLoss_weight', default=-1, type=float,
                        help="weight hyperparameter for specific generatorloss")
    parser.add_argument('--IdLoss_weight', default=-1, type=float,
                        help="weight hyperparameter for specific generatorloss")

    # data arguments
    parser.add_argument('--batch_size', type=int, default=DEBUG_BATCH_SIZE, help='Batch size to run trainer.')

    return parser.parse_args()


def manipulate_defaults_for_own_test(args):
    """
    function to manipulate the parsed arguments quickly so we don't lose the actual defaults or run by terminal

    :return:
    """

    # args.epochs = 5  # etc..
    pass


if __name__ == '__main__':
    print("cuda_version", torch.version.cuda, "pytorch version", torch.__version__, "python version", sys.version)
    ensure_current_directory()
    args = parse()
    manipulate_defaults_for_own_test(args)
    main(args)
