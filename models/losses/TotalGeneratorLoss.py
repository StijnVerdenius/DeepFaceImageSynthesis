import torch

from models.discriminators.PatchDiscriminator import PatchDiscriminator
from models.generators.pix2pixGenerator import pix2pixGenerator
from models.losses.GeneralLoss import GeneralLoss
from models.losses.PerceptualLoss import PerceptualLoss
from models.losses.TripleConsistencyLoss import TripleConsistencyLoss
from models.losses.pix2pixGLoss import pix2pixGLoss


class TotalGeneratorLoss(GeneralLoss):

    def __init__(self, pp_weight=10, adv_weight=1, trip_weight=100, id_weight=1, self_weight = 100, pix_weight = 10, **kwargs):
        super().__init__()
        self.pp = PerceptualLoss(pp_weight)
        self.adv = pix2pixGLoss(adv_weight)
        self.trip = TripleConsistencyLoss(trip_weight)

        # todo: add @ elias

    def forward(self, imgs, generated_imgs, landmarks_real, in_between_landmarks, target_landmarks, generator, discriminator):
        """ combined loss function from the tiple-cons paper """

        loss_pp = self.pp.forward(imgs, generated_imgs)
        loss_triple = self.trip.forward(imgs, in_between_landmarks, target_landmarks, generator)
        loss_adv = self.adv(generated_imgs, discriminator)

        return loss_pp + loss_adv + loss_triple # todo: add @elias


if __name__ == '__main__':
    z = TotalGeneratorLoss()

    testinput = torch.rand((20, 3, 28, 28))
    testgenerated = torch.rand((20, 3, 28, 28))
    test_landmarks_real = torch.rand((20, 68, 28, 28))
    test_landmarks_in_between = torch.rand((20, 68, 28, 28))
    test_landmarks_targets = torch.rand((20, 68, 28, 28))

    G = pix2pixGenerator()
    D = PatchDiscriminator()

    bana = z.forward(testinput, testgenerated, test_landmarks_real, test_landmarks_in_between, test_landmarks_targets, G, D)

    print(bana.shape, bana)



