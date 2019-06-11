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
from testing.test import test
import torch.optim as opt
import torch
from data.Dataset300VW import X300VWDataset
import numpy as np
import sys

torch.backends.cudnn.benchmark = True

def dummy_batch(batch_size, channels):
    return np.random.normal(0, 1, (batch_size, channels, IMSIZE, IMSIZE))


def load_data(keyword: str, batch_size: int) -> DataLoader:  # todo @ klaus

    data = None

    if (keyword == "train"):
        data = DataLoader(X300VWDataset(), shuffle=True, batch_size=batch_size, drop_last=True) #Changed to false!!!
    elif (keyword == "validate"):
        data = DataLoader(X300VWDataset(), shuffle=True, batch_size=batch_size, drop_last=True) #Changed to false!!!
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
    dataloader_train = load_data("train", arguments.batch_size)
    dataloader_validate = load_data("validate", arguments.batch_size)

    # get models
    embedder = find_right_model(EMBED_DIR, arguments.embedder,
                                device=DEVICE,
                                n_channels_in=INPUT_SIZE,
                                n_channels_out=arguments.embedding_size).to(DEVICE)

    generator = find_right_model(GEN_DIR, arguments.generator,
                                 device=DEVICE,
                                 n_channels_in=INPUT_SIZE).to(DEVICE)

    discriminator = find_right_model(DIS_DIR, arguments.discriminator,
                                     device=DEVICE,
                                     n_channels_in=INPUT_SIZE).to(DEVICE)

    # assertions
    assert_type(GeneralGenerator, generator)
    assert_type(GeneralDiscriminator, discriminator)
    assert_type(GeneralEmbedder, embedder)

    # train or test
    if (arguments.mode == "train" or arguments.mode == "finetune"):

        # init optimizers
        generator_optimizer = opt.Adam(generator.parameters(), arguments.learning_rate)
        discriminator_optimizer = opt.Adam(discriminator.parameters(), arguments.learning_rate)
        embedder_optimizer = opt.Adam(embedder.parameters(), arguments.learning_rate)

        # define loss functions
        loss_gen = find_right_model(LOSS_DIR, arguments.loss_gen, weight=arguments.weight_advloss)
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
        test(dataloader_validate, embedder, generator, discriminator, arguments)

    else:

        raise Exception(f"Unrecognized train/test mode?: {arguments.mode}")

    if (arguments.timing):
        stop_timing(pr)


def parse():
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--epochs', default=5, type=int, help='max number of epochs') ##################### SHOULD BE 100!!! changed it for DEBUGGING!
    parser.add_argument('--eval_freq', type=int, default=5, help='Frequency (batch-wise) of evaluation')
    parser.add_argument('--plot_freq', type=int, default=25, help='Frequency (batch-wise) of plotting pictures')
    parser.add_argument('--saving_freq', type=int, default=10, help='Frequency (epoch-wise) of saving models')
    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--feedback', default=False, type=bool, help='whether to plot or not during training')
    parser.add_argument('--mode', default="train", type=str, help="'train', 'test' or 'finetune'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    # debug
    parser.add_argument('--timing', type=bool, default=False, help='are we measuring efficiency?')

    # test arguments
    parser.add_argument('--test_model_date', default="", type=str, help='date_stamp string for which model to load')
    parser.add_argument('--test_model_suffix', default="", type=str, help='filename string for which model to load')

    # model arguments
    parser.add_argument('--embedding_size', default=2, type=int, help='dimensionality of latent embedding space')
    parser.add_argument('--embedder', default="EmptyEmbedder", type=str, help="name of objectclass")
    parser.add_argument('--discriminator', default="PatchDiscriminator", type=str, help="name of objectclass")
    parser.add_argument('--generator', default="ResnetGenerator", type=str, help="name of objectclass")

    # loss arguments
    parser.add_argument('--loss_gen', default="NonSaturatingGLoss", type=str, help="name of objectclass")
    parser.add_argument('--loss_dis', default="DefaultDLoss", type=str, help="name of objectclass")

    # hyperparams
    parser.add_argument('--weight_advloss', default=1, type=int, help="name of objectclass")
    parser.add_argument('--weight_triploss', default=1, type=int, help="name of objectclass")
    parser.add_argument('--weight_pploss', default=1, type=int, help="name of objectclass")

    # data arguments
    parser.add_argument('--batch_size', type=int, default=DEBUG_BATCH_SIZE, help='Batch size to run trainer.')
    # todo @ klaus

    return parser.parse_args()


def manipulate_defaults_for_own_test(args):
    """
    function to manipulate the parsed arguments quickly so we don't lose the actual defaults or run by terminal

    :return:
    """

    # args.epochs = 5  # etc..
    pass


if __name__ == '__main__':
    print("cuda_version" , torch.version.cuda, "pytorch version", torch.__version__, "python version", sys.version)
    ensure_current_directory()
    args = parse()
    manipulate_defaults_for_own_test(args)
    main(args)
