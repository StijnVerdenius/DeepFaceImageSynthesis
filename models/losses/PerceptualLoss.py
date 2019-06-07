from models.generators.GeneralGenerator import GeneralGenerator
from models.generators.pix2pixGenerator import pix2pixGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from typing import Tuple
from utils.constants import *
from utils.model_utils import *


class PerceptualLoss(GeneralLoss):

    def __init__(self, weight, feature_selection=(3, 8, 15, 24), **kwargs):
        self.feature_selection = feature_selection
        super(PerceptualLoss, self).__init__(weight=weight)

    def custom_forward(self, batch: torch.Tensor, generated_images: torch.Tensor):
        """ forward pass """

        # get vgg feats
        real_feats, fake_feats = self.get_features(batch, generated_images)

        # reconstruction loss
        l1_part = 0
        for real, fake in zip(real_feats, fake_feats):
            l1_part += L1_distance(real, fake)

        # style loss
        frobenius = self.frobenius_norm(self.gram_matrix(real_feats[2]), self.gram_matrix(fake_feats[2]))

        return frobenius + l1_part

    def frobenius_norm(self, batch_1: torch.Tensor, batch_2: torch.Tensor):
        """ forbenius norm (just normal norm?)"""

        return torch.norm((batch_1 - batch_2))  # todo: revisit

    def gram_matrix(self, batch: torch.Tensor):
        """ calculates gram matrix """

        # dims
        a, b, c, d = batch.size()

        # make 2d (batch * ch) x (h * w)
        features = batch.view((a * b, c * d))

        # get gram-matrix
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps. todo: is this necessary?
        return G.div(a * b * c * d)

    def get_features(self, batch: torch.Tensor, generated_images: torch.Tensor, ) \
            -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ]:
        """ extracts features from vgg network """

        usable_feats = VGG.features[:self.feature_selection[-1] + 1]

        feat_1_2 = usable_feats[:self.feature_selection[0] + 1]
        feat_2_2 = usable_feats[self.feature_selection[0] + 1:self.feature_selection[1] + 1]
        feat_3_3 = usable_feats[self.feature_selection[1] + 1:self.feature_selection[2] + 1]
        feat_4_3 = usable_feats[self.feature_selection[2] + 1:]

        a_1_2_real = feat_1_2.forward(batch)
        a_1_2_fake = feat_1_2.forward(generated_images)

        a_2_2_real = feat_2_2.forward(a_1_2_real)
        a_2_2_fake = feat_2_2.forward(a_1_2_fake)

        a_3_3_real = feat_3_3.forward(a_2_2_real)
        a_3_3_fake = feat_3_3.forward(a_2_2_fake)

        a_4_3_real = feat_4_3.forward(a_3_3_real)
        a_4_3_fake = feat_4_3.forward(a_3_3_fake)

        return (a_1_2_real, a_2_2_real, a_3_3_real, a_4_3_real), (a_1_2_fake, a_2_2_fake, a_3_3_fake, a_4_3_fake)


if __name__ == '__main__':
    z = PerceptualLoss(1)

    testinput = torch.rand((20, 3, 28, 28))
    testinput_2 = torch.rand((20, 3, 28, 28))

    bana = z.forward(testinput, testinput_2)

    print(bana.shape, bana)
