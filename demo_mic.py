
import cv2
from dataclasses import dataclass, asdict
import glob
import numpy as np
import os
import pyaudio
import scipy.io as sio
from scipy.io import wavfile
import shutil
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import webrtcvad

from mfcc import MFCC
from config import NETWORKS_PARAMETERS
from network import get_network, SynergyNet
from utils import voice2face, read_obj
from vad import read_wave, write_wave, frame_generator, vad_collector
from pyaudio_recording import Recorder
from utilf.render import render_vert

@dataclass
class StreamParams:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    frames_per_buffer: int = 1024
    input: bool = True
    output: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def rm_sil(voice_file, vad_obj):
    """
    remove silence
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def get_fbank(voice, mfc_obj):
    """
    process audio and create mel-spectrogram
    """
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank

def voice2face(e_net, g_net, voice_file, vad_obj, mfc_obj, GPU=True):
    vad_voice = rm_sil(voice_file, vad_obj)
    fbank = get_fbank(vad_voice, mfc_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net(embedding)
    return face

def main():
    # recording and save under the root
    filename = "audio.wav"
    # stream_params = StreamParams()
    # recorder = Recorder(stream_params)
    # # record for 5 seconds
    # recorder.record(5, filename)

    # initialization
    # voice activity detector, aggressiveness = 2
    vad_obj = webrtcvad.Vad(2)
    # Mel-Frequency extractor
    mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)
    # net definition
    e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False)
    g_net, _ = get_network('g', NETWORKS_PARAMETERS, train=False)

    # building models: unsupervised
    image3D = SynergyNet(pretrained=False, last_CN=None).cuda().eval()
    backbone_ckpt = torch.load(NETWORKS_PARAMETERS['image3D']['model_path'])
    image3D.load_state_dict(backbone_ckpt)

    # SynergyNet pretrained network for getting pose
    image3D_pretrained = SynergyNet(pretrained=True).cuda().eval()

    # data and config
    up_layer = torch.nn.Upsample((120,120), mode='bilinear', align_corners=True)
    tri = sio.loadmat('./train.configs/tri.mat')['tri']

    # default savepath
    FOLDER_ROOT = 'data/results/'
    if not os.path.exists(FOLDER_ROOT):
        os.makedirs(FOLDER_ROOT)

    with torch.no_grad():
        # voice2face
        face_image = voice2face(e_net, g_net, filename, vad_obj, mfc_obj, NETWORKS_PARAMETERS['GPU'])
        face_image =  up_layer(face_image)

        # Pose from 3DDFA-V2
        pose = image3D_pretrained(face_image, return_onlypose=True)
        R, off = image3D_pretrained.parse_param_102_pose(pose)            

        # Alignment with synthesized image
        prediction_fr = image3D(face_image)
        prediction = R @ prediction_fr + off

    # calculation between mean male and female shape and classify the gender by meshes
    #print(prediction_fr.requires_grad)
    prediction_fr_np = prediction_fr.squeeze(0).cpu().numpy()
    prediction_fr_np = np.transpose(prediction_fr_np, (1,0))
    mean_male = read_obj('male.obj') # 53215 * 3
    mean_female = read_obj('female.obj') # 53215 * 3
    N_vertices = prediction_fr_np.shape[0] #53215
    error_male = np.linalg.norm(prediction_fr_np - mean_male)/ N_vertices
    error_female = np.linalg.norm(prediction_fr_np - mean_female)/ N_vertices
    
    pred_midD = np.linalg.norm(prediction_fr_np[2130]-prediction_fr_np[15003])
    pred_foreD = np.linalg.norm(prediction_fr_np[1678]-prediction_fr_np[42117])
    pred_cheekD = np.linalg.norm(prediction_fr_np[2294]-prediction_fr_np[13635])
    pred_earD = np.linalg.norm(prediction_fr_np[20636]-prediction_fr_np[34153])
    print("-------------------------")
    if error_male < error_female:
        print("This is a male's voice")
        print("Statistics from the predicted mesh and mean gender mesh")
        target_foreD = np.linalg.norm(mean_male[1678]-mean_male[42117])
        target_cheekD = np.linalg.norm(mean_male[2294]-mean_male[13635])
        target_earD = np.linalg.norm(mean_male[20636]-mean_male[34153])
        target_midD = np.linalg.norm(mean_male[2130]-mean_male[15003])

        ratio_fore = (pred_foreD-target_foreD)/target_foreD
        ratio_cheek = (pred_cheekD-target_cheekD)/target_cheekD
        ratio_ear = (pred_earD-target_earD)/target_earD
        ratio_mid = (pred_midD-target_midD)/target_midD

        print(f"The forehead is {ratio_fore*100}% than the mean male shape")
        print(f"The cheek-to-cheek is {ratio_cheek*100}% than the mean male shape")
        print(f"The ear-to-ear is {ratio_ear*100}% than the mean male shape")
        print(f"The midline is {ratio_mid*100}% than the mean male shape")
    else:
        print("This is a female's voice")
        print("Statistics from the predicted mesh and mean gender mesh")
        target_foreD = np.linalg.norm(mean_female[1678]-mean_female[42117])
        target_cheekD = np.linalg.norm(mean_female[2294]-mean_female[13635])
        target_earD = np.linalg.norm(mean_female[20636]-mean_female[34153])
        target_midD = np.linalg.norm(mean_female[2130]-mean_female[15003])

        ratio_fore = (pred_foreD-target_foreD)/target_foreD
        ratio_cheek = (pred_cheekD-target_cheekD)/target_cheekD
        ratio_ear = (pred_earD-target_earD)/target_earD
        ratio_mid = (pred_midD-target_midD)/target_midD

        print(f"The forehead is {ratio_fore*100}% than the mean female shape")
        print(f"The cheek-to-cheek is {ratio_cheek*100}% than the femean male shape")
        print(f"The ear-to-ear is {ratio_ear*100}% than the mean female shape")
        print(f"The midline is {ratio_mid*100}% than the mean female shape")
    print("-------------------------")

    wide_shape = read_obj('wide.obj')
    skinny_shape = read_obj('skinny.obj')
    regular_shape = read_obj('regular.obj')
    slim_shape = read_obj('slim.obj')
    error_wide = np.linalg.norm(prediction_fr_np - wide_shape)/ N_vertices
    error_skinny = np.linalg.norm(prediction_fr_np - skinny_shape)/ N_vertices
    error_regular = np.linalg.norm(prediction_fr_np - regular_shape)/ N_vertices
    error_slim = np.linalg.norm(prediction_fr_np - slim_shape)/ N_vertices
    err_type = np.array([error_wide, error_skinny, error_regular, error_slim])
    index = np.argsort(err_type)[0]

    if index == 0:
        print("The face shape is closer to WIDE")
    elif index == 1:
        print(f"The face shape is closer to SKINNY")
    elif index == 2:
        print(f"The face shape is closer to REGULAR")
    elif index == 3:
        print(f"The face shape is closer to SLIM")

    print("-------------------------")

    # transform to image coordinate space
    prediction[:, 1, :] = 127 - prediction[:, 1, :]
    save_name = os.path.join(FOLDER_ROOT, 'micIn')
    img = (((face_image[0].clamp(-1,1))*127.5)+128).detach().cpu().numpy().astype(np.uint8)
    img = np.transpose(img, (1,2,0))
    img = img[:,:,[2,1,0]]
    pred = prediction[0].detach().cpu().numpy()

    # save
    cv2.imwrite(save_name+'_image.png', img)
    render_vert(img, pred, alpha=1.0, wfp=save_name+'_overlap.png')

    vutils.save_image(face_image.detach().clamp(-1,1), filename.replace('.wav', '.png'), normalize=True)


if __name__ == '__main__':
    main()