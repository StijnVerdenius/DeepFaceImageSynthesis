from models.generators.GeneralGenerator import GeneralGenerator
from models.generators.pix2pixGenerator import pix2pixGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch


class TripleConsistencyLoss(GeneralLoss):

    def __init__(self, weight, **kwargs):
        super(TripleConsistencyLoss, self).__init__(weight=weight)

    def custom_forward(self,
                batch: torch.Tensor,
                in_between_landmarks: torch.Tensor,
                target_landmarks: torch.Tensor,
                generator: GeneralGenerator):

        direct_output = generator.forward(torch.cat((batch, target_landmarks), dim=1))

        in_between_output = generator.forward(torch.cat((batch, in_between_landmarks), dim=1))

        indirect_output = generator.forward(torch.cat((in_between_output, target_landmarks), dim=1))

        norm = torch.sum((indirect_output-direct_output).pow(2), dim=(1,2,3)) # todo: revisit correctness

        return norm.mean()


if __name__ == '__main__':

    z = TripleConsistencyLoss(1)

    testinput = torch.rand((20, 3, 28, 28))
    landm = torch.rand((20, 68, 28, 28))
    landm1 = torch.rand((20, 68, 28, 28))

    bana = z.forward(testinput, landm, landm1,  pix2pixGenerator())

    print(bana.shape, bana)
