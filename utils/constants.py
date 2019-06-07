from models.general.data_management import DataManager
import torch
import torchvision.models.vgg as vgg

PIC_DIR = "output_pictures"
MODELS_DIR = "saved_models"
PROGRESS_DIR = "training_progress"
LOSS_DIR = "losses"
EMBED_DIR = "embedders"
GEN_DIR = "generators"
DIS_DIR = "discriminators"
IMSIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PREFIX_OUTPUT = "results/output"
DATA_MANAGER = DataManager(f"./{PREFIX_OUTPUT}/")
CODE_DIR = "codebase"
OUTPUT_DIRS = [PIC_DIR, PROGRESS_DIR, MODELS_DIR, CODE_DIR]
VGG = vgg.vgg19(pretrained=True)
VGG.eval()
CHANNEL_DIM = 1