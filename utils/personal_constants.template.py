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
DATASET_PERSON_OUTPUT_PATH = Path(f'PATH_TO_DATASETS/datasets/person_processed_dim{constants.DATASET_300VW_IMSIZE}')

# download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
DLIB_PREDICTOR_PATH = Path('./data/local_data/shape_predictor_68_face_landmarks.dat')

check_paths = [DATASET_300VW_OUTPUT_PATH, DLIB_PREDICTOR_PATH, DATASET_PERSON_OUTPUT_PATH]
assert all([p.exists() for p in check_paths])
WRITER_DIRECTORY = "results/output"
