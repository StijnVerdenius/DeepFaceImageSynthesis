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

from utils.constants import DEVICE, CHANNEL_DIM
from utils.training_helpers import unpack_batch


class TotalGeneratorLoss(GeneralLoss):

    def __init__(self, PerceptualLoss_weight: float = 10, NonSaturatingGLoss_weight: float = 1,
                 TripleConsistencyLoss_weight: float = 100, IdLoss_weight: float = 1,
                 ConsistencyLoss_weight: float = 100, PixelLoss_weight: float = 10,
                 **kwargs):
        super().__init__()
        self.pp = PerceptualLoss(PerceptualLoss_weight)
        self.adv = NonSaturatingGLoss(NonSaturatingGLoss_weight)
        self.trip = TripleConsistencyLoss(TripleConsistencyLoss_weight)
        self.self = ConsistencyLoss(ConsistencyLoss_weight)
        self.pix = PixelLoss(PixelLoss_weight)
        self.id = IdLoss(IdLoss_weight)

    def forward(self, generator: GeneralGenerator,
                discriminator: GeneralDiscriminator,
                batch_1: Dict[str, torch.Tensor],
                batch_2: Dict[str, torch.Tensor],
                batch_3: Dict[str, torch.Tensor]) \
            -> Tuple[
                torch.Tensor, Dict, torch.Tensor, torch.Tensor, torch.Tensor
            ]:
        """ combined loss function from the tiple-cons paper """

        # prepare input
        image_1, landmarks_1 = unpack_batch(batch_1)
        image_2, landmarks_2 = unpack_batch(batch_2)
        _, landmarks_3 = unpack_batch(batch_3)
        image_1 = image_1.to(DEVICE).float()
        image_2 = image_2.to(DEVICE).float()
        landmarks_2 = landmarks_2.to(DEVICE).float()

        target_landmarked_input = torch.cat((image_1, landmarks_2), dim=CHANNEL_DIM)
        target_landmarked_truth = torch.cat((image_2, landmarks_2), dim=CHANNEL_DIM)
        fake = generator.forward(target_landmarked_input)
        target_landmarked_fake = torch.cat((fake, landmarks_2), dim=CHANNEL_DIM)

        # adverserial loss
        loss_adv, save_adv = self.adv(target_landmarked_fake, discriminator)

        # style losses
        loss_pp, save_pp = self.pp.forward(image_1, fake)  # todo: which images to use
        loss_id, save_id = self.id(image_1, fake)
        loss_pix, save_pix = self.pix(image_1, fake)

        # consistency losses
        loss_triple, save_triple = self.trip.forward(image_1, fake, landmarks_3, landmarks_2, generator)
        loss_self, save_self = self.self(image_1, fake, landmarks_1, generator)

        # get total loss
        total = loss_pp + loss_adv + loss_triple + loss_pix + loss_self + loss_id

        # merge dicts
        merged = {**save_adv, **save_pix, **save_pp, **save_self, **save_triple, **save_id}

        return total, merged, fake, target_landmarked_fake, target_landmarked_truth

if __name__ == '__main__':


    loss_func = TotalGeneratorLoss()

    testinput1 = torch.rand((20, 3, 28, 28))
    testinput2 = torch.rand((20, 3, 28, 28))
    testinput3 = torch.rand((20, 3, 28, 28))

    testlandmarks1 = torch.rand((20, 68, 28, 28))
    testlandmarks2 = torch.rand((20, 68, 28, 28))
    testlandmarks3 = torch.rand((20, 68, 28, 28))

    batch1 = {"image" : testinput1, "landmarks" : testlandmarks1}
    batch2 = {"image" : testinput2, "landmarks" : testlandmarks2}
    batch3 = {"image" : testinput3, "landmarks" : testlandmarks3}

    G = ResnetGenerator(n_channels_in=71).to(DEVICE)
    D = PatchDiscriminator(n_channels_in=71).to(DEVICE).eval()

    bana = loss_func.forward(G, D, batch1, batch2, batch3)

    print(bana[0].shape, bana[0], bana[1])
