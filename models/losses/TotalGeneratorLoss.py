import torch

from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.generators.GeneralGenerator import GeneralGenerator
from models.discriminators.PatchDiscriminator import PatchDiscriminator
from models.generators.ResnetGenerator import ResnetGenerator
from models.losses.GeneralLoss import GeneralLoss
from models.losses.PerceptualLoss import PerceptualLoss
from models.losses.TripleConsistencyLoss import TripleConsistencyLoss
from models.losses.NonSaturatingGLoss import NonSaturatingGLoss
from models.losses.ConsistencyLoss import ConsistencyLoss
from models.losses.PixelLoss import PixelLoss
from models.losses.IdLoss import IdLoss

from typing import Tuple, Dict


class TotalGeneratorLoss(GeneralLoss):

    def __init__(self, pp_weight:float=10, adv_weight:float=1, trip_weight:float=100, id_weight:float=1, self_weight:float=100, pix_weight:float=10,
                 **kwargs):
        super().__init__()
        self.pp = PerceptualLoss(pp_weight)
        self.adv = NonSaturatingGLoss(adv_weight)
        self.trip = TripleConsistencyLoss(trip_weight)
        self.self = ConsistencyLoss(self_weight)
        self.pix = PixelLoss(pix_weight)
        self.id = IdLoss(id_weight)


    def forward(self, imgs:torch.Tensor, generated_imgs:torch.Tensor, landmarks_real: torch.Tensor, in_between_landmarks: torch.Tensor, target_landmarks:torch.Tensor,
                generator: GeneralGenerator, discriminator: GeneralDiscriminator):
        """ combined loss function from the tiple-cons paper """

        loss_pp, save_pp = self.pp.forward(imgs, generated_imgs)
        loss_triple, save_triple = self.trip.forward(imgs, in_between_landmarks, target_landmarks, generator)
        loss_adv, save_adv = self.adv(generated_imgs, discriminator)
        loss_self, save_self = self.self(imgs, landmarks_real, target_landmarks, generator)
        loss_pix, save_pix = self.pix(imgs, target_landmarks, generator)
        loss_id, save_id = self.id(imgs, generated_imgs)

        # get total loss
        total =  loss_pp + loss_adv + loss_triple + loss_pix + loss_self + loss_id

        # merge dicts
        merged = {**save_adv, **save_pix, **save_pp, **save_self, **save_triple, **save_id}

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
                     G, D)[0]

    print(bana.shape, bana)
