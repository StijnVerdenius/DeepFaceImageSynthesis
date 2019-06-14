from models.generators.GeneralGenerator import GeneralGenerator
from models.generators.ResnetGenerator import ResnetGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch

from utils.constants import DEVICE
from utils.training_helpers import CHANNEL_DIM, L2_distance, L1_distance


class TripleConsistencyLoss(GeneralLoss):

    def __init__(self, weight: float, **kwargs):
        super(TripleConsistencyLoss, self).__init__(weight=weight)

    def custom_forward(self,
                       batch: torch.Tensor, # input images (non landmarked)
                       direct_output: torch.Tensor, # fake generations (non landmarked)
                       in_between_landmarks: torch.Tensor,
                       target_landmarks: torch.Tensor,
                       generator: GeneralGenerator):

        in_between_landmarks = in_between_landmarks.to(DEVICE).float()

        in_between_output = generator.forward(torch.cat((batch, in_between_landmarks), dim=CHANNEL_DIM))

        indirect_output = generator.forward(torch.cat((in_between_output, target_landmarks), dim=CHANNEL_DIM))

        # l1 norm
        norm = L1_distance(indirect_output, direct_output).pow(2)

        in_between_landmarks.detach()
        in_between_output.detach()
        indirect_output.detach()

        return norm.mean()


if __name__ == '__main__':
    z = TripleConsistencyLoss(1)
