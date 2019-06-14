from enum import Enum

import cv2
import torch
import torchvision.models.vgg as vgg

from models.general.data_management import DataManager

# directories
PIC_DIR = "output_pictures"
MODELS_DIR = "saved_models"
PROGRESS_DIR = "training_progress"
LOSS_DIR = "losses"
EMBED_DIR = "embedders"
GEN_DIR = "generators"
DIS_DIR = "discriminators"
PREFIX_OUTPUT = "results/output"
CODE_DIR = "codebase"
OUTPUT_DIRS = [PIC_DIR, PROGRESS_DIR, MODELS_DIR, CODE_DIR]

# data manager

# train
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMSIZE = 64
CHANNEL_DIM = 1
INPUT_LANDMARK_CHANNELS = 68
INPUT_CHANNELS = 3
INPUT_SIZE = INPUT_CHANNELS + INPUT_LANDMARK_CHANNELS
DEBUG_BATCH_SIZE = 8
TOTAL_LOSS = "TotalGeneratorLoss"
DATA_MANAGER = DataManager(f"./{PREFIX_OUTPUT}/")

# vgg
VGG = vgg.vgg19(pretrained=True)
VGG = VGG.to(DEVICE)
VGG.eval()

# printing
PRINTCOLOR_PURPLE = '\033[95m'
PRINTCOLOR_CYAN = '\033[96m'
PRINTCOLOR_DARKCYAN = '\033[36m'
PRINTCOLOR_BLUE = '\033[94m'
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_YELLOW = '\033[93m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_UNDERLINE = '\033[4m'
PRINTCOLOR_END = '\033[0m'

# dataset
INTERPOLATION = cv2.INTER_CUBIC
DATASET_300VW_IMSIZE = 128
DATASET_300VW_PADDING = 0
DATASET_300VW_EXPAND_RATIO = 2.0
DATASET_300VW_LIMIT_N_VIDEOS = None
DATASET_300VW_N_VIDEOS = 114
DATASET_300VW_N_LANDMARKS = 68
DATASET_300VW_ANNOTATIONS_INPUT_FOLDER = 'annot'
DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION = 'pts'
DATASET_300VW_VIDEO_FILE_NAME = 'vid.avi'
DATASET_300VW_IMAGES_TEMP_FOLDER = 'images'
DATASET_300VW_IMAGES_TEMP_EXTENSION = 'jpg'
DATASET_300VW_IMAGES_OUTPUT_FOLDER = 'images'
DATASET_300VW_IMAGES_OUTPUT_EXTENSION = 'jpg'
DATASET_300VW_IMAGE_QUALITY = 100
DATASET_300VW_OUTER_LOOP_DESCRIPTION = 'video'
DATASET_300VW_INNER_LOOP_DESCRIPTION = 'frame'
DATASET_300VW_NUMBER_FORMAT = '06d'


class Dataset300VWMode(Enum):
    TEST_1 = [
        '114',
        '124',
        '125',
        '126',
        '150',
        '158',
        '401',
        '402',
        '505',
        '506',
        '507',
        '508',
        '509',
        '510',
        '511',
        '514',
        '515',
        '518',
        '519',
        '520',
        '521',
        '522',
        '524',
        '525',
        '537',
        '538',
        '540',
        '541',
        '546',
        '547',
        '548',
    ]
    TEST_2 = [
        '203',
        '208',
        '211',
        '212',
        '213',
        '214',
        '218',
        '224',
        '403',
        '404',
        '405',
        '406',
        '407',
        '408',
        '409',
        '412',
        '550',
        '551',
        '553',
    ]
    TEST_3 = [
        '410',
        '411',
        '516',
        '517',
        '526',
        '528',
        '529',
        '530',
        '531',
        '533',
        '557',
        '558',
        '559',
        '562',
    ]
    TRAIN = [
        '001',
        '002',
        '003',
        '004',
        '007',
        '009',
        '010',
        '011',
        '013',
        '015',
        '016',
        '017',
        '018',
        '019',
        '020',
        '022',
        '025',
        '027',
        '028',
        '029',
        '031',
        '033',
        '034',
        '035',
        '037',
        '039',
        '041',
        '043',
        '044',
        '046',
        '047',
        '048',
        '049',
        '053',
        '057',
        '059',
        '112',
        '113',
        '115',
        '119',
        '120',
        '123',
        '138',
        '143',
        '144',
        '160',
        '204',
        '205',
        '223',
        '225',
    ]
    ALL = sorted(TEST_1 + TEST_2 + TEST_3 + TRAIN)


assert len(Dataset300VWMode.ALL.value) == DATASET_300VW_N_VIDEOS

