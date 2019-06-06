from utils.general_utils import ensure_current_directory
from utils.model_utils import find_right_model, load_models_and_state
from utils.constants import *
import argparse
from training.train import train
from testing.test import test
import torch.optim as opt
import torch


def main(arguments):
    # data
    dataloader_train = [None, None]  # todo @ klaus
    dataloader_validate = [None, None]  # todo @ klaus

    # determine input size
    input_size = 2  # todo @ klaus

    # get models
    embedder = find_right_model(EMBED_DIR, arguments.embedder,
                                device=DEVICE,
                                n_input=input_size,
                                n_output=arguments.embedding_size)

    generator = find_right_model(GEN_DIR, arguments.generator,
                                 device=DEVICE,
                                 n_input=input_size)

    discriminator = find_right_model(DIS_DIR, arguments.discriminator,
                                     device=DEVICE,
                                     n_input=input_size)

    # train or test
    if (arguments.mode == "train" or arguments.mode == "finetune"):

        # init optimizers
        generator_optimizer = opt.Adam(generator.parameters(), arguments.learning_rate)
        discriminator_optimizer = opt.Adam(discriminator.parameters(), arguments.learning_rate)

        loss_gen = find_right_model(LOSS_DIR, arguments.loss_gen)
        loss_dis = find_right_model(LOSS_DIR, arguments.loss_dis)

        # train
        trained_succesfully = train(dataloader_train,
                                    dataloader_validate,
                                    loss_gen,
                                    loss_dis,
                                    embedder,
                                    generator,
                                    discriminator,
                                    arguments,
                                    discriminator_optimizer,
                                    generator_optimizer)

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

        raise Exception("Unrecognized train/test mode")


def parse():
    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--epochs', default=50, type=int, help='max number of epochs')
    parser.add_argument('--device', default="cpu", type=str, help='device')
    parser.add_argument('--feedback', default=False, type=bool, help='whether to plot or not during training')
    parser.add_argument('--mode', default="train", type=str, help="'train', 'test' or 'finetune'")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval_freq', type=int, default=200, help='Frequency of evaluation on the test set')
    parser.add_argument('--plot_freq', type=int, default=200, help='Frequency of evaluation on the test set')

    # test arguments
    parser.add_argument('--test_model_date', default="", type=str, help='date_stamp string for which model to load')
    parser.add_argument('--test_model_suffix', default="", type=str, help='filename string for which model to load')

    # model arguments
    parser.add_argument('--embedding_size', default=2, type=int, help='dimensionality of latent embedding space')
    parser.add_argument('--embedder', default="InitialEmbedder", type=str, help="name of objectclass")
    parser.add_argument('--discriminator', default="PatchDiscriminator", type=str, help="name of objectclass") ##### SET TO PATCH GAN
    parser.add_argument('--generator', default="pix2pixGenerator", type=str, help="name of objectclass") ########### SET TO PIX2PIX

    # loss arguments
    parser.add_argument('--loss_gen', default="pix2pixGLoss", type=str, help="name of objectclass")  ########### SET TO PIX2PIX
    parser.add_argument('--loss_dis', default="pix2pixDLoss", type=str, help="name of objectclass") ########### SET TO PIX2PIX

    # data arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size to run trainer.')
    # todo @ klaus

    return parser.parse_args()


def manipulate_defaults_for_own_test(args):
    """
    function to manipulate the parsed arguments quickly so we don't lose the actual defaults or run by terminal

    :return:
    """

    args.epochs = 100  # etc..


if __name__ == '__main__':
    ensure_current_directory()
    args = parse()
    manipulate_defaults_for_own_test(args)
    main(args)
