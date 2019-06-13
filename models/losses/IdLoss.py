from models.generators.GeneralGenerator import GeneralGenerator
from models.generators.ResnetGenerator import ResnetGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from typing import Tuple
from utils.constants import *
# from utils.model_utils import
from utils.training_helpers import L1_distance
from typing import Tuple, Dict

class IdLoss(GeneralLoss):

    def __init__(self, weight: float, feature_selection: Tuple=(13), **kwargs): ####Use ReLU 2.3 - layer 13!!!!!!!!!!!!!!!!!!
        self.feature_selection = feature_selection
        super(IdLoss, self).__init__(weight=weight)

    def custom_forward(self, batch: torch.Tensor, generated_images: torch.Tensor):
        """ forward pass """

        # get vgg feats
        real_feats, fake_feats = self.get_features(batch, generated_images)

        # reconstruction loss
        l1 = 0
        for real, fake in zip(real_feats, fake_feats):
            l1 += L1_distance(real, fake).mean()

        return l1

    def get_features(self, batch: torch.Tensor, generated_images: torch.Tensor) \
            -> Tuple[
                torch.Tensor,
                torch.Tensor
            ]:
        """ extracts features from vgg network """

        usable_feats = VGG.features

        feat_r2_3 = usable_feats[:self.feature_selection + 1]


        a_r2_3_real = feat_r2_3.forward(batch)
        a_r2_3_fake = feat_r2_3.forward(generated_images)

        return (a_r2_3_real), (a_r2_3_fake)


if __name__ == '__main__':
    z = IdLoss(1)

    testinput = torch.rand((20, 3, 28, 28))
    testinput_2 = torch.rand((20, 3, 28, 28))

    bana = z.forward(testinput, testinput_2)[0]

    print(bana.shape,bana)
