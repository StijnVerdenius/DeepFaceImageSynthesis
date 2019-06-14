from pathlib import Path

from utils import constants


raise Exception('update paths for personal use please, then remove this exception')
DATASET_300VW_RAW_PATH = Path('PATH_TO_DATASETS/datasets/300VW_Dataset_raw')
DATASET_300VW_TEMP_PATH = Path('PATH_TO_DATASETS/datasets/300VW_Dataset_temp')
DATASET_300VW_OUTPUT_PATH = Path(f'PATH_TO_DATASETS/datasets/300VW_Dataset_processed_dim{constants.IMSIZE}')
