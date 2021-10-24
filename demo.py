# This script is for testing.
import os
import glob
import torch
import scipy.io as sio
import numpy as np
import cv2

from config import NETWORKS_PARAMETERS
from network import get_network, IGM
from utils import voice2face_processed
from utilf.render import render_vert

# initialization
e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False)
g_net, _ = get_network('g', NETWORKS_PARAMETERS, train=False)

# building models: Voice2Mesh unsupervised
image3D = IGM(pretrained=False, last_CN=None).cuda().eval()
backbone_ckpt = torch.load(NETWORKS_PARAMETERS['image3D']['model_path'])
image3D.load_state_dict(backbone_ckpt)

# 3DDFA-V2 pretrained network for getting pose
image3D_pretrained = IGM(pretrained=True).cuda().eval()

# data and config
voice_list = sorted(glob.glob('data/preprocessed_MFCC/*'))
up_layer = torch.nn.Upsample((120,120), mode='bilinear', align_corners=True)
tri = sio.loadmat('./train.configs/tri.mat')['tri']

# [TODO] Change this variable to yout result output folder
FOLDER_ROOT = 'data/results/'

if not os.path.exists(FOLDER_ROOT):
    os.mkdir(FOLDER_ROOT)

for folder in voice_list:
    index = folder.rsplit('/',1)[-1]
    print(index)

    if not os.path.exists(FOLDER_ROOT+index):
        os.mkdir(FOLDER_ROOT + index)

    all_sequences = sorted(glob.glob(folder+'/*'))
    for sequence in all_sequences:
        all_fbanks = sorted(glob.glob(sequence+'/*.npy'))
        sequence_name = sequence.rsplit('/',1)[-1]
        
        for fbank in all_fbanks:
            fbank_name = fbank.rsplit('/',1)[-1][:-4]
            face_image = voice2face_processed(e_net, g_net, fbank,
                                    NETWORKS_PARAMETERS['GPU'])               
            face_image =  up_layer(face_image)

            # Pose from 3DDFA-V2
            pose = image3D_pretrained(face_image, return_onlypose=True)
            R, off = image3D_pretrained.parse_param_102_pose(pose)            
            
            #Alignment with synthesized image
            prediction = image3D(face_image)
            prediction = R @ prediction + off

            # transform to image coordinate space
            prediction[:, 1, :] = 127 - prediction[:, 1, :]
            save_name = FOLDER_ROOT+ index + '/' + sequence_name + '_' + fbank_name
            img = (((face_image[0].clamp(-1,1))*127.5)+128).detach().cpu().numpy().astype(np.uint8)
            img = np.transpose(img, (1,2,0))
            img = img[:,:,[2,1,0]]
            pred = prediction[0].detach().cpu().numpy()
            # save
            cv2.imwrite(save_name+'_image.png', img)
            render_vert(img, pred, alpha=1.0, wfp=save_name+'_overlap.png')

