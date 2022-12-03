#from dataset import VoiceDataset, FaceDataset
from network import VoiceEmbedNet, Generator
from utils import get_collate_fn
import os

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
