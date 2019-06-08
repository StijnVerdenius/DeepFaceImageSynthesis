import torch

from models.discriminators.PatchDiscriminator import PatchDiscriminator
from models.generators.ResnetGenerator import ResnetGenerator
from models.losses.GeneralLoss import GeneralLoss
from models.losses.PerceptualLoss import PerceptualLoss
from models.losses.TripleConsistencyLoss import TripleConsistencyLoss
from models.losses.NonSaturatingGLoss import NonSaturatingGLoss
from models.losses.ConsistencyLoss import ConsistencyLoss
from models.losses.PixelLoss import PixelLoss

from typing import Tuple, Dict


class TotalGeneratorLoss(GeneralLoss):

    def __init__(self, pp_weight=10, adv_weight=1, trip_weight=100, id_weight=1, self_weight=100, pix_weight=10,
                 **kwargs):
        super().__init__()
        self.pp = PerceptualLoss(pp_weight)
        self.adv = NonSaturatingGLoss(adv_weight)
        self.trip = TripleConsistencyLoss(trip_weight)
        self.self = ConsistencyLoss(self_weight)
        self.pix = PixelLoss(pix_weight)

        # NEED TO ADD ID LOSS

        # todo: add @ elias

    def forward(self, imgs, generated_imgs, landmarks_real, in_between_landmarks, target_landmarks, generator,
                discriminator):
        """ combined loss function from the tiple-cons paper """

        loss_pp, save_pp = self.pp.forward(imgs, generated_imgs)
        loss_triple, save_triple = self.trip.forward(imgs, in_between_landmarks, target_landmarks, generator)
        loss_adv, save_adv = self.adv(generated_imgs, discriminator)
        loss_self, save_self = self.self(imgs, landmarks_real, target_landmarks, generator)
        loss_pix, save_pix = self.pix(imgs, target_landmarks, generator)

        # get total loss
        total =  loss_pp + loss_adv + loss_triple + loss_pix + loss_self

        # merge dicts
        merged = {**save_adv, **save_pix, **save_pp, **save_self, **save_triple}

        # return
        return total, merged

if __name__ == '__main__':
    z = TotalGeneratorLoss()

    testinput = torch.rand((20, 3, 28, 28))
    testgenerated = torch.rand((20, 3, 28, 28))
    test_landmarks_real = torch.rand((20, 68, 28, 28))
    test_landmarks_in_between = torch.rand((20, 68, 28, 28))
    test_landmarks_targets = torch.rand((20, 68, 28, 28))

    G = ResnetGenerator()
    D = PatchDiscriminator()

    bana = z.forward(testinput, testgenerated, test_landmarks_real, test_landmarks_in_between, test_landmarks_targets,
                     G, D)

    print(bana.shape, bana)
