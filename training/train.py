from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.general.statistic import Statistic
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from models.losses.TotalGeneratorLoss import TotalGeneratorLoss
from testing.test import plot_batch
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
from tensorboardX import SummaryWriter  ####### TESTING tensorboard


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
                 loss_gen: TotalGeneratorLoss,
                 loss_dis: GeneralLoss,
                 arguments):

        DATA_MANAGER.set_date_stamp()

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

        # initialize tensorboardx
        # self.writer = SummaryWriter(
        #     f"results/output/{DATA_MANAGER.stamp}/tensorboardx/"
        self.writer = SummaryWriter(
            f"/home/lgpu0293/ProjectAI/DeepFakes/results/output/tensorboardx/{DATA_MANAGER.stamp}/tensorboardx/")

        self.labels_train = None
        self.labels_validate = None

        self.plotting_batch_1, self.plotting_batch_2, self.plotting_batch_3 = next(iter(self.dataloader_validation))

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
                        batch_1: Dict,
                        batch_2: Dict,
                        batch_3: Dict,
                        train=True,
                        accuracy_discriminator =0) \
            -> Tuple[Dict, Dict, torch.Tensor, int]:
        """
         inner loop of epoch iteration

        """

        if (train):
            # set generator to train and discriminator to evaluation
            self.trainer_gen.prepare_training()
            self.trainer_dis.prepare_evaluation()

        # forward pass generator
        loss_gen, loss_gen_saving, fake, landmarked_fake, landmarked_truth = self.loss_gen.forward(self.generator,
                                                                                                   self.discriminator,
                                                                                                   batch_1,
                                                                                                   batch_2,
                                                                                                   batch_3)
        if (train):
            # backward pass generator
            self.trainer_gen.do_backward(loss_gen)

            # set discriminator to train
            self.trainer_dis.prepare_training()
            self.trainer_gen.prepare_evaluation()

        # combine real and fake
        if (self.labels_train is None and train):
            self.labels_train = torch.cat((torch.zeros(fake.shape[0]).to(DEVICE), torch.ones(fake.shape[0]).to(DEVICE)),
                                          dim=0)  # combine real and fake
        if (self.labels_validate is None and not train):
            self.labels_validate = torch.cat(
                (torch.zeros(fake.shape[0]).to(DEVICE), torch.ones(fake.shape[0]).to(DEVICE)),
                dim=0)

        usable_labels = self.labels_train if (train) else self.labels_validate

        landmarked_truth = landmarked_truth.detach()
        landmarked_fake = landmarked_fake.detach()

        # forward pass discriminator
        predictions_true = self.discriminator.forward(landmarked_truth.detach())
        predictions_fake = self.discriminator.forward(landmarked_fake.detach())
        predictions = torch.cat((predictions_fake, predictions_true), dim=0)

        loss_dis, loss_dis_saving = self.loss_dis.forward(predictions, usable_labels)

        if (train):

            if accuracy_discriminator < self.arguments.DiscAccCap:
                # backward discriminator
                self.trainer_dis.do_backward(loss_dis)

        accuracy_discriminator = calculate_accuracy(predictions, self.labels_train)

        # detaching
        fake = fake.detach()
        predictions = predictions.detach()
        loss_gen = loss_gen.detach()
        loss_dis = loss_dis.detach()
        landmarked_truth = landmarked_truth.detach()
        landmarked_fake = landmarked_fake.detach()

        # print flush
        sys.stdout.flush()

        return loss_gen_saving, loss_dis_saving, fake, accuracy_discriminator

    def epoch_iteration(self, epoch_num: int) -> List[Statistic]:
        """
        one epoch implementation

        """

        progress = []

        for i, (batch_1, batch_2, batch_3) in enumerate(self.dataloader_train):

            # run batch iteration
            loss_gen, loss_dis, fake_images, accuracy_discriminator = self.batch_iteration(batch_1, batch_2, batch_3)

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

                # log to terminal and retrieve a statistics object
                statistic = self.log(loss_gen_actual, loss_dis_actual, loss_gen, loss_dis, batches_passed,
                                     accuracy_discriminator)

                # assert type
                assert_type(Statistic, statistic)

                # append statistic to list
                progress.append(statistic)

                time_passed = datetime.now() - DATA_MANAGER.actual_date

                if (
                        (time_passed.total_seconds() > (
                                self.arguments.max_training_minutes * 60)) and self.arguments.max_training_minutes > 0):
                    raise KeyboardInterrupt(
                        f"Process killed because {self.arguments.max_training_minutes} minutes passed since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")

            # save a set of pictures
            if (batches_passed % self.arguments.plot_freq == 0):
                _, _, example_images, _ = self.batch_iteration(self.plotting_batch_1, self.plotting_batch_2,
                                                               self.plotting_batch_3, train=False)

                example_images = example_images.view(-1, 3, IMSIZE, IMSIZE)
                example_images = BGR2RGB_pytorch(example_images)
                plot_some_pictures(example_images, batches_passed)
                self.writer.add_image('fake_samples', vutils.make_grid(example_images, normalize=True),
                                      batches_passed)

                big_image = plot_batch(self.plotting_batch_1, self.plotting_batch_2, self.plotting_batch_3,
                                       self.embedder, self.generator, self.arguments, number_of_pictures=3)
                big_image = torch.from_numpy(np.moveaxis(big_image, -1, 0)).float()

                self.writer.add_image('landmark_comparison', vutils.make_grid(big_image, normalize=True), batches_passed,)

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
            loss_gen, loss_dis, _, accuracy_discriminator = self.batch_iteration(batch_1, batch_2, batch_3,
                                                                                 train=False)

            # convert dicts to ints
            loss_gen_actual = sum(loss_gen.values())
            loss_dis_actual = sum(loss_dis.values())

            # assertions
            assert_type(float, loss_gen_actual)
            assert_type(float, loss_dis_actual)
            assert_type(float, accuracy_discriminator)

            # append findings to respective lists
            total_accuracy.append(accuracy_discriminator)
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

        # pass stats to tensorboardX
        for e in list(loss_gen_dict.keys()):
            self.writer.add_scalar(f'loss/gen/{e}', loss_gen_dict[e], batches_passed)
        for e in list(loss_dis_dict.keys()):
            self.writer.add_scalar(f'loss/dis/{e}', loss_dis_dict[e], batches_passed)

        self.writer.add_scalar("loss/gen/total", loss_gen, batches_passed)
        self.writer.add_scalar("accuracy/dis", discriminator_accuracy, batches_passed)

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
            f"details: {loss_gen_dict}",
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


        except KeyboardInterrupt as e:
            print(f"Killed by user: {e}")
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
