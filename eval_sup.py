# This script is for batch processing testing.

import os
import glob
import torch
import torchvision.utils as vutils
import webrtcvad
import scipy.io as sio
import csv
import numpy as np

from mfcc import MFCC
from config import NETWORKS_PARAMETERS
from network import get_network, Generator1D_directMLP
from utils import write_obj_with_colors, voice2face_processed_MeshOut

# initialization
vad_obj = webrtcvad.Vad(2)
mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)
e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False)

g_net = Generator1D_directMLP().cuda().eval()
g_net_ckpt = torch.load(NETWORKS_PARAMETERS['g']['model_path'])
g_net.load_state_dict(g_net_ckpt)

# test
voice_list = sorted(glob.glob('data/fbank/*'))
up_layer = torch.nn.Upsample((120,120), mode='bilinear', align_corners=True)
tri = sio.loadmat('./train.configs/tri.mat')['tri']

id_name = {}
csv_file = open('data/vox1_meta.csv')
rows=csv.reader(csv_file, delimiter='	')
headers = next(rows)
for row in rows:
    id_name.update({row[0]:row[1]})
available_GT = list(map(lambda k: k.rsplit('/',1)[-1], sorted(glob.glob('A2E_val/*'))))

# [TODO] Change this variable to yout result output folder
FOLDER_ROOT = 'data/supervised_output/'

if not os.path.exists(FOLDER_ROOT):
    os.mkdir(FOLDER_ROOT)
coll = []
for folder in voice_list:
    index = folder.rsplit('/',1)[-1]
    print(index)
    if index > 'id10309': # The end of E is 10309
        break
    corr_name = id_name[index]
    if not corr_name in available_GT: #check if the fbank id is in the fitted model database
        continue
    count = 0

    if not os.path.exists(FOLDER_ROOT+corr_name):
        os.mkdir(FOLDER_ROOT + corr_name)

    all_sequences = sorted(glob.glob(folder+'/*'))
    
    for sequence in all_sequences:
        all_fbanks = sorted(glob.glob(sequence+'/*.npy'))
        sequence_name = sequence.rsplit('/',1)[-1]
        
        for fbank in all_fbanks:
            fbank_name = fbank.rsplit('/',1)[-1][:-4]
            prediction = voice2face_processed_MeshOut(e_net, g_net, fbank,NETWORKS_PARAMETERS['GPU']).squeeze().detach().cpu()
            save_name = FOLDER_ROOT+ corr_name + '/' + sequence_name + '_' + fbank_name
            write_obj_with_colors(save_name+'.obj', prediction, triangles=tri)

            count += 1
            # the first three in all the fbank sequences
            if count >= 3:
                break

        if count >= 3:
            break

