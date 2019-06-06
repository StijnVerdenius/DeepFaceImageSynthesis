import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders.GeneralEmbedder import GeneralEmbedder
from models.generators.GeneralGenerator import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
from training.train import TrainingProcess
from utils.general_utils import ensure_current_directory


# todo: override methods from regular training for meta training

class MetaTrain(TrainingProcess):

    def __init__(self, generator: GeneralGenerator,
                 discriminator: GeneralDiscriminator,
                 embedder: GeneralEmbedder,
                 dataloader_train: DataLoader,
                 dataloader_validation: DataLoader,
                 optimizer_gen: Optimizer,
                 optimizer_dis: Optimizer,
                 optimizer_emb: Optimizer,
                 loss_gen: GeneralLoss,
                 loss_dis: GeneralLoss,
                 arguments,
                 costum_field_example=None):

        super().__init__(generator, discriminator, embedder, dataloader_train, dataloader_validation, optimizer_gen,
                         optimizer_dis, optimizer_emb, loss_gen, loss_dis, arguments)

        self.costum_field = costum_field_example  # example

    def new_method_example(self):
        print(self.costum_field)  # example new method

    def batch_iteration(self, batch: torch.Tensor, landmarks: torch.Tensor, train=True):
        # example overriding parent method

        return 1, 1, None, None, None


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
