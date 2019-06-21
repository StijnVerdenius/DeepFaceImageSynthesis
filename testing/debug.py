from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data import transformations
from data.Dataset300VW import X300VWDataset
from models.generators.UNetGenerator import UNetGenerator
from utils.constants import *
from utils.general_utils import ensure_current_directory

import torch

ensure_current_directory()

DATA_MANAGER.directory = "./data/local_data/"

gen = UNetGenerator(use_dropout=True, n_hidden=64)

generator_state = DATA_MANAGER.load_python_obj("banana")["generator"]

gen.load_state_dict(generator_state)

transform = transforms.Compose(
    [
        transformations.RandomHorizontalFlip(),
        transformations.RandomCrop(probability=0.5),
        transformations.Resize(),
        transformations.RescaleValues(),
        transformations.ChangeChannels(),
    ]
)

data = DataLoader(X300VWDataset(Dataset300VWMode.TEST_1, transform=transform, n_videos_limit=1),
                  shuffle=True, batch_size=5, drop_last=True)

gen.to(DEVICE)

gen.eval()

b1, b2, _ = next(iter(data))

imgs1 = b1["image"].float()
lm1 = b1["landmarks"].float()
imgs1 = torch.cat((imgs1, lm1), dim=1)

imgs2 = b2["image"].float()
lm2 = b2["landmarks"].float()
imgs2 = torch.cat((imgs2, lm2), dim=1)

first = imgs1
second = torch.cat((imgs1[:4], imgs2[3:4]), dim=0)

a = gen.forward(first.to(DEVICE))
b = gen.forward(second.to(DEVICE))

print(torch.allclose(a[:4], b[:4]))
