import torch
import torchvision.models.vgg as vgg

from models.general.data_management import DataManager

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
INPUT_LANDMARK_CHANNELS = 68
INPUT_CHANNELS = 3
INPUT_SIZE = INPUT_CHANNELS + INPUT_LANDMARK_CHANNELS
CHANNEL_DIM = 1
DATASET_300VW_PADDING = 0
DATASET_300VW_EXPAND_RATIO = 2.0
DATASET_300VW_LIMIT_N_VIDEOS = 5
DATASET_300VW_N_VIDEOS = 114
DATASET_300VW_N_LANDMARKS = 68
DATASET_300VW_ANNOTATIONS_INPUT_FOLDER = 'annot'
DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION = 'pts'
DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER = 'annotations'
DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION = 'txt'
DATASET_300VW_VIDEO_FILE_NAME = 'vid.avi'
DATASET_300VW_IMAGES_TEMP_FOLDER = 'images'
DATASET_300VW_IMAGES_TEMP_EXTENSION = 'jpg'
DATASET_300VW_IMAGES_OUTPUT_FOLDER = 'images'
DATASET_300VW_IMAGES_OUTPUT_EXTENSION = 'jpg'
DATASET_300VW_IMAGE_QUALITY = 100
DATASET_300VW_OUTER_LOOP_DESCRIPTION = 'video'
DATASET_300VW_INNER_LOOP_DESCRIPTION = 'frame'
DATASET_300VW_NUMBER_FORMAT = '06d'
