from pathlib import Path

from utils import constants


# please create a copy of this file named 'personal_constants.py',
# remove the exception below and update the paths.
# Only DATASET_300VW_OUTPUT_PATH is necessary for training the network.
# The other two variables are needed for processing the dataset.
raise Exception('This is just a template')
DATASET_300VW_RAW_PATH = Path('PATH_TO_DATASETS/datasets/300VW_Dataset_raw')
DATASET_300VW_TEMP_PATH = Path('PATH_TO_DATASETS/datasets/300VW_Dataset_temp')
DATASET_300VW_OUTPUT_PATH = Path(f'PATH_TO_DATASETS/datasets/300VW_Dataset_processed_dim{constants.DATASET_300VW_IMSIZE}')
WRITER_DIRECTORY = "results/output"
