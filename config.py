import string
from dataset import VoiceDataset, FaceDataset
from network import VoiceEmbedNet, Generator, FaceEmbedNet, Classifier
from utils import get_collate_fn, get_collate_fn_4
import os

DATASET_PARAMETERS = {
    # meta data provided by voxceleb1 dataset
    'meta_file': 'data/vox1_meta.csv',

    # voice dataset
    'voice_dir': 'data/fbank',
    'voice_ext': 'npy',

    # face dataset
    'face_dir': 'data/VGG_ALL_FRONTAL',
    'face_ext': '.jpg',

    # train data includes the identities
    # whose names start with the characters of 'FGH...XYZ' 
    'split': string.ascii_uppercase[5:],

    # dataloader
    'voice_dataset': VoiceDataset,
    'face_dataset': FaceDataset,
    'batch_size': 64,
    'nframe_range': [300, 800],
    'workers_num': 1,
    'collate_fn': get_collate_fn,
    'collate_fn_4': get_collate_fn_4,

    # test data
    'test_data': 'data/test_data/'
}

SAVE_DIR = 'pretrained_models/'
NUM_EPOCH = 48000 #49999

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

NETWORKS_PARAMETERS = {
    
    'SAVE_DIR': SAVE_DIR,

    # VOICE EMBEDDING NETWORK (e)
    'e': {
        'network': VoiceEmbedNet,
        'input_channel': 64,
        'channels': [256, 384, 576, 864],
        'output_channel': 64, # the embedding dimension
        'model_path': 'pretrained_models/voice_embedding.pth',
    },
    # GENERATOR (g)
    'g': {
        'network': Generator,
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64], # channels for deconvolutional layers
        'output_channel': 3, # images with RGB channels
        'model_path': f'{SAVE_DIR}/generator_{NUM_EPOCH}.pth'
    },
    # FACE EMBEDDING NETWORK (f)
    'f': {
        'network': FaceEmbedNet,
        'input_channel': 3,
        'channels': [32, 64, 128, 256, 512],
        'output_channel': 64,
        'model_path': 'models/face_embedding.pth',
    },
    # DISCRIMINATOR (d)
    'd': {
        'network': Classifier, # Discrminator is a special Classifier with 1 subject
        'input_channel': 64,
        'channels': [],
        'output_channel': 1,
        'model_path': 'models/discriminator.pth',
    },
    # CLASSIFIER (c)
    'c': {
        'network': Classifier,
        'input_channel': 64,
        'channels': [],
        'output_channel': -1, # This parameter is depended on the dataset we used
        'model_path': 'models/classifier.pth',
    },
    # OPTIMIZER PARAMETERS 
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,

    # MODE, use GPU or not
    'GPU': True,

    'image3D':{
        'model_path': f'{SAVE_DIR}/image3D_{NUM_EPOCH}.pth'
    }
}
