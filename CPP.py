import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.utils_cpp import cpp_predictor
from utils.utils_cpp import cpp_generator
from utils.utils_cpp import cpp_optimizer
from utils.utils_common.activator import Activation

GENERATOR_DATA_PATH = './dataset/data_cpp/cpp_generator_dataset.txt'
GENERATOR_MODEL_PATH = './model/model_cpp/cpp_generator.hdf5'
SEED_SEQ_LENGTH = 10

PREDICTOR_DATA_PATH = './dataset/data_cpp/cpp_predictor_dataset.csv'
PREDICTOR_MODEL_PATH = './model/model_cpp/cpp_predictor.hdf5'
PREDICTOR_STATS_PATH = './dataset/data_cpp/cpp_predictor_dataset_stats.json'

SMILES_PATH = './dataset/data_cpp/cpp_smiles.json'
FP_RADIUS = 3
FP_BITS = 1024
SEQ_MAX = 108

generator = cpp_generator.Generator(data_path = GENERATOR_DATA_PATH, seq_length = SEED_SEQ_LENGTH)

generator.train_model(
    model_params = {
        'epochs': 2,
        'save_checkpoint': True,
        'checkpoint_filepath': './model/'
    }
)

predictor = cpp_predictor.Predictor(
        data_path = PREDICTOR_DATA_PATH,
        smiles_path = SMILES_PATH,
        fp_radius = FP_RADIUS,
        fp_bits = FP_BITS,
        seq_max = SEQ_MAX
)

predictor.train_model(
    model_params = {
        'epochs': 2,
        'save_checkpoint': True,
        'checkpoint_filepath': './model/'
    }
)

optimizer = cpp_optimizer.Optimizer(
        model_path = PREDICTOR_MODEL_PATH,
        data_path = PREDICTOR_DATA_PATH,
        smiles_path = SMILES_PATH,
        stats_path = PREDICTOR_STATS_PATH,
        fp_radius = FP_RADIUS,
        fp_bits = FP_BITS,
        seq_max = SEQ_MAX
)

generator = cpp_generator.Generator(
        model_path = GENERATOR_MODEL_PATH,
        data_path = GENERATOR_DATA_PATH,
        seq_length = SEED_SEQ_LENGTH
)

list_seeds = generator.generate_seed(n_seeds = 2, seed_length = 30)

df = optimizer.optimize(list_seeds)
df.head(2)


activator = Activation(
        mode = 'cpp',
        model_path = PREDICTOR_MODEL_PATH,
        smiles_path = SMILES_PATH,
        stats_path = PREDICTOR_STATS_PATH,
        fp_radius = FP_RADIUS,
        fp_bits = FP_BITS,
        seq_max = SEQ_MAX
)

activator.analyze('RQIKIWFQNRRMKWKK')


