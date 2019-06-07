from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.statistic import Statistic
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from utils.general_utils import *
from utils.constants import *
from models.general.trainer import Trainer
from utils.model_utils import save_models
import random
import torch
from utils.training_helpers import *
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class TrainingProcess:

    def __init__(self,
                 generator: GeneralGenerator,
                 discriminator: GeneralDiscriminator,
                 embedder: GeneralEmbedder,
                 dataloader_train: DataLoader,
                 dataloader_validation: DataLoader,
                 optimizer_gen: Optimizer,
                 optimizer_dis: Optimizer,
                 optimizer_emb: Optimizer,
                 loss_gen: GeneralLoss,
                 loss_dis: GeneralLoss,
                 arguments):

        # models
        self.generator = generator
        self.discriminator = discriminator
        self.embedder = embedder

        # data
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation

        # optimizers
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.optimizer_emb = optimizer_emb

        # loss functions
        self.loss_gen = loss_gen
        self.loss_dis = loss_dis
        self.arguments = arguments

        # trainers
        self.trainer_gen = Trainer([generator], [optimizer_gen])
        self.trainer_dis = Trainer([discriminator], [optimizer_dis])
        self.trainer_emb = Trainer([embedder], [optimizer_emb])

        # for combining batches
        self.combined_batch_size = 2 * arguments.batch_size
        self.shuffle_indices = list(range(int(self.combined_batch_size)))

        # assert type
        assert_type(GeneralGenerator, generator)
        assert_type(GeneralEmbedder, embedder)
        assert_type(GeneralDiscriminator, discriminator)
        assert_type(GeneralLoss, loss_dis)
        assert_type(GeneralLoss, loss_gen)

        # assert nonzero
        assert_non_empty(arguments)
        assert_non_empty(optimizer_dis)
        assert_non_empty(optimizer_emb)
        assert_non_empty(optimizer_gen)
        assert_non_empty(self.shuffle_indices)
        assert_non_empty(dataloader_train)
        assert_non_empty(dataloader_validation)

    def batch_iteration(self, batch: torch.Tensor, landmarks: torch.Tensor, train=True) \
            -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         inner loop of epoch iteration

        """

        # prepare input
        batch.to(DEVICE)
        landmarks.to(DEVICE)
        landmarked_batch = torch.cat((batch, landmarks), dim=CHANNEL_DIM)

        if (train):
            # set generator to train and discriminator to evaluation
            self.trainer_gen.prepare_training()
            self.trainer_dis.prepare_evaluation()

        # forward pass generator
        fake = self.generator.forward(landmarked_batch)
        loss_gen = self.loss_gen.forward(fake, self.discriminator)

        if (train):
            # backward pass generator
            self.trainer_gen.do_backward(loss_gen)

            # set discriminator to train
            self.trainer_dis.prepare_training()

        # combine real and fake
        landmarked_fake = torch.cat((fake, landmarks), dim=CHANNEL_DIM)
        combined_set, labels = combine_real_and_fake(self.shuffle_indices, landmarked_batch, landmarked_fake)

        # forward pass discriminator
        predictions = self.discriminator.forward(combined_set)
        loss_dis = self.loss_dis.forward(predictions, labels)

        if (train):
            # backward discriminator
            self.trainer_dis.do_backward(loss_dis)

        fake.detach()
        predictions.detach()
        labels.detach()

        return loss_gen.item(), loss_dis.item(), fake, predictions, labels

    def epoch_iteration(self, epoch_num: int) -> List[Statistic]:
        """
        one epoch implementation

        """

        progress = []

        for i, (batch, landmarks) in enumerate(self.dataloader_train):  # todo: how to split the data @ Klaus

            # run batch iteration
            loss_gen, loss_dis, fake_images, _, _ = self.batch_iteration(batch, landmarks)

            # assertions
            assert_type(int, loss_gen)
            assert_type(int, loss_dis)

            # calculate amount of passed batches
            batches_passed = i + (epoch_num * len(self.dataloader_train))

            # print progress to terminal
            if (batches_passed % self.arguments.eval_freq == 0):
                # log to terminal and retrieve a statistics object
                statistic = self.log(loss_gen, loss_dis)

                # assert type
                assert_type(Statistic, statistic)

                # append statistic to list
                progress.append(statistic)

            # save a set of pictures
            if (batches_passed % self.arguments.plot_freq == 0):
                plot_some_pictures(self.arguments.feedback, fake_images, batches_passed)

        return progress

    def validate(self) -> Tuple[int, int, int]:
        """
        Runs a validation epoch

        """

        # init
        total_loss_discriminator = []
        total_loss_generator = []
        total_accuracy = []

        for i, (batch, landmarks) in self.dataloader_validation:  # todo: how to split? @klaus
            # run batch iteration
            loss_gen, loss_dis, _, predictions, actual_labels = self.batch_iteration(batch, landmarks, train=False)

            # also get accuracy
            accuracy = calculate_accuracy(predictions, actual_labels)

            # assertions
            assert_type(int, loss_gen)
            assert_type(int, loss_dis)
            assert_type(int, accuracy)

            # append findings to respective lists
            total_accuracy.append(accuracy)
            total_loss_discriminator.append(loss_dis)
            total_loss_discriminator.append(loss_gen)

        return mean(total_loss_generator), mean(total_loss_discriminator), mean(total_accuracy)

    def log(self, loss_gen: int, loss_dis: int) -> Statistic:
        """
        logs to terminal and calculate log_statistics

        """

        # put models in evaluation mode
        self.trainer_dis.prepare_evaluation()
        self.trainer_gen.prepare_evaluation()

        # validate on validationset
        loss_gen_validate, loss_dis_validate, discriminator_accuracy = self.validate()

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

    def train(self) -> bool:
        """
         main training function

        :return:
        """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self.arguments)

        # data gathering
        progress = []

        try:

            # run
            for epoch in range(self.arguments.epochs):
                # do epoch
                epoch_progress = self.epoch_iteration(epoch)

                # add progress
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                DATA_MANAGER.save_python_obj(progress, f"{DATA_MANAGER.stamp}/{PROGRESS_DIR}/progress_list")

                # write models if needed (don't save the first one
                if (epoch + 1 % self.arguments.saving_freq == 0):
                    save_models(self.discriminator, self.generator, self.embedder, f"Models_at_epoch_{epoch}")

        except KeyboardInterrupt:
            print("Killed by user")
            save_models(self.discriminator, self.generator, self.embedder, f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            save_models(self.discriminator, self.generator, self.embedder, f"CRASH_at_epoch_{epoch}")
            raise e

        # example last save
        save_models(self.discriminator, self.generator, self.embedder, "finished")
        return True


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
