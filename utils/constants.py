from models.general.data_management import DataManager
import torch

PIC_DIR = "output_pictures"
MODELS_DIR = "saved_models"
PROGRESS_DIR = "training_progress"
DATA_MANAGER = DataManager("./results/")
LOSS_DIR = "losses"
EMBED_DIR = "embedders"
GEN_DIR = "generators"
DIS_DIR = "discriminators"
IMSIZE = 100  # todo: change @ Klaus
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
