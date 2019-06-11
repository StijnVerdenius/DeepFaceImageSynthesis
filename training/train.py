from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.statistic import Statistic
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from utils.general_utils import *
from utils.constants import *
from models.general.trainer import Trainer
from utils.model_utils import save_models
from utils.training_helpers import *
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
from datetime import datetime
import sys
import os


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

        self.labels = None

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

    def batch_iteration(self,
                        batch_1: torch.Tensor,
                        batch_2: torch.Tensor,
                        batch_3: torch.Tensor,
                        train=True) \
            -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         inner loop of epoch iteration

        """

        # prepare input
        image_1, landmarks_1 = unpack_batch(batch_1)
        image_2, landmarks_2 = unpack_batch(batch_2)
        image_3, landmarks_3 = unpack_batch(batch_3)
        image_1 = image_1.to(DEVICE).float()
        image_2 = image_2.to(DEVICE).float()
        image_3 = image_3.to(DEVICE).float()
        landmarks_1 = landmarks_1.to(DEVICE).float()
        landmarks_2 = landmarks_2.to(DEVICE).float()
        landmarks_3 = landmarks_3.to(DEVICE).float()
        target_landmarked_batch = torch.cat((image_1, landmarks_2), dim=CHANNEL_DIM)
        truth_landmarked_batch = torch.cat((image_2, landmarks_2), dim=CHANNEL_DIM)

        if (train):
            # set generator to train and discriminator to evaluation
            self.trainer_gen.prepare_training()
            self.trainer_dis.prepare_evaluation()

        # forward pass generator
        fake = self.generator.forward(target_landmarked_batch)
        landmarked_fake = torch.cat((fake, landmarks_2), dim=CHANNEL_DIM)
        loss_gen, loss_gen_saving = self.loss_gen.forward(landmarked_fake, self.discriminator)

        if (train):
            # backward pass generator
            self.trainer_gen.do_backward(loss_gen)

            # set discriminator to train
            self.trainer_dis.prepare_training()
            self.trainer_gen.prepare_evaluation()

        # combine real and fake
        if (self.labels is None):
            self.labels = torch.cat((torch.zeros(fake.shape[0]).to(DEVICE), torch.ones(image_1.shape[0]).to(DEVICE)))

        combined_set, labels = combine_real_and_fake(self.shuffle_indices, truth_landmarked_batch.detach(),
                                                     landmarked_fake.detach(), self.labels)

        # forward pass discriminator
        predictions = self.discriminator.forward(combined_set.detach())
        loss_dis, loss_dis_saving = self.loss_dis.forward(predictions, labels)

        if (train):
            # backward discriminator
            self.trainer_dis.do_backward(loss_dis)

        # detaching
        fake = fake.detach().cpu()
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()
        image_1 = image_1.detach().cpu()
        image_2 = image_2.detach().cpu()
        image_3 = image_3.detach().cpu()
        landmarks_1 = landmarks_1.detach().cpu()
        landmarks_2 = landmarks_2.detach().cpu()
        landmarks_3 = landmarks_3.detach().cpu()
        loss_gen = loss_gen.detach().cpu()
        loss_dis = loss_dis.detach().cpu()
        target_landmarked_batch = target_landmarked_batch.detach().cpu()
        truth_landmarked_batch = truth_landmarked_batch.detach().cpu()
        landmarked_fake = landmarked_fake.detach().cpu()

        # print flush
        sys.stdout.flush()

        return loss_gen_saving, loss_dis_saving, fake, predictions, labels

    def epoch_iteration(self, epoch_num: int) -> List[Statistic]:
        """
        one epoch implementation

        """

        progress = []

        for i, (batch_1, batch_2, batch_3) in enumerate(self.dataloader_train):

            if i > 20:  ###### ADDED THIS FOR DEBUGGING!!!!!!!
                break
            # else:

            # run batch iteration
            loss_gen, loss_dis, fake_images, predictions, labels = self.batch_iteration(batch_1, batch_2, batch_3)

            # assertions
            assert_type(dict, loss_gen)
            assert_type(dict, loss_dis)

            # calculate amount of passed batches
            batches_passed = i + (epoch_num * len(self.dataloader_train))

            # print progress to terminal
            if (batches_passed % self.arguments.eval_freq == 0):
                # convert dicts to ints
                loss_gen_actual = sum(loss_gen.values())
                loss_dis_actual = sum(loss_dis.values())
                accuracy_discriminator = calculate_accuracy(predictions, labels)

                # log to terminal and retrieve a statistics object
                statistic = self.log(loss_gen_actual, loss_dis_actual, loss_gen, loss_dis, batches_passed,
                                     accuracy_discriminator)

                # assert type
                assert_type(Statistic, statistic)

                # append statistic to list
                progress.append(statistic)

            # save a set of pictures
            if (batches_passed % self.arguments.plot_freq == 0):
                plot_some_pictures(self.arguments.feedback, fake_images, batches_passed)

            # empty cache
            torch.cuda.empty_cache()

        return progress

    def validate(self) -> Tuple[float, float, float]:
        """
        Runs a validation epoch

        """

        # init
        total_loss_discriminator = []
        total_loss_generator = []
        total_accuracy = []

        for i, (batch_1, batch_2, batch_3) in enumerate(self.dataloader_validation):
            # run batch iteration
            loss_gen, loss_dis, _, predictions, actual_labels = self.batch_iteration(batch_1, batch_2, batch_3,
                                                                                     train=False)

            # also get accuracy
            accuracy = calculate_accuracy(predictions, actual_labels)

            # convert dicts to ints
            loss_gen_actual = sum(loss_gen.values())
            loss_dis_actual = sum(loss_dis.values())

            # assertions
            assert_type(float, loss_gen_actual)
            assert_type(float, loss_dis_actual)
            assert_type(float, accuracy)

            # append findings to respective lists
            total_accuracy.append(accuracy)
            total_loss_generator.append(loss_gen_actual)
            total_loss_discriminator.append(loss_dis_actual)

            # empty cache
            torch.cuda.empty_cache()

        return mean(total_loss_generator), mean(total_loss_discriminator), mean(total_accuracy)

    def log(self, loss_gen: float, loss_dis: float, loss_gen_dict: Dict, loss_dis_dict: Dict,
            batches_passed: int, discriminator_accuracy: float) -> Statistic:
        """
        logs to terminal and calculate log_statistics

        """

        # put models in evaluation mode
        self.trainer_dis.prepare_evaluation()
        self.trainer_gen.prepare_evaluation()

        # validate on validationset
        loss_gen_validate, loss_dis_validate, _ = 0, 0, 0  # self.validate() todo: do we want to restore this?

        stat = Statistic(loss_gen_train=loss_gen,
                         loss_dis_train=loss_dis,
                         loss_gen_val=loss_gen_validate,
                         loss_dis_val=loss_dis_validate,
                         loss_gen_train_dict=loss_gen_dict,
                         loss_dis_train_dict=loss_dis_dict,
                         dis_acc=discriminator_accuracy)

        # print in-place with 3 decimals
        print(
            f"\r",
            f"batch: {batches_passed}/{len(self.dataloader_train)}",
            f"|\t {stat}",
            end='')

        # define training statistic
        return stat

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

            print(f"{PRINTCOLOR_BOLD}Started training with the following config:{PRINTCOLOR_END}\n{self.arguments}")

            # run
            for epoch in range(self.arguments.epochs):

                print(
                    f"\n\n{PRINTCOLOR_BOLD}Starting epoch{PRINTCOLOR_END} {epoch}/{self.arguments.epochs} at {str(datetime.now())}")

                # do epoch
                epoch_progress = self.epoch_iteration(epoch)

                # add progress
                progress += epoch_progress

                # write progress to pickle file (overwrite because there is no point keeping seperate versions)
                DATA_MANAGER.save_python_obj(progress, f"{DATA_MANAGER.stamp}/{PROGRESS_DIR}/progress_list",
                                             print_success=False)

                # write models if needed (don't save the first one
                if (((epoch + 1) % self.arguments.saving_freq) == 0):
                    save_models(self.discriminator, self.generator, self.embedder, f"Models_at_epoch_{epoch}")

                # flush prints
                sys.stdout.flush()


        except KeyboardInterrupt:
            print("Killed by user")
            save_models(self.discriminator, self.generator, self.embedder, f"KILLED_at_epoch_{epoch}")
            return False
        except Exception as e:
            print(e)
            save_models(self.discriminator, self.generator, self.embedder, f"CRASH_at_epoch_{epoch}")
            raise e

        # flush prints
        sys.stdout.flush()

        # example last save
        save_models(self.discriminator, self.generator, self.embedder, "finished")
        return True


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
