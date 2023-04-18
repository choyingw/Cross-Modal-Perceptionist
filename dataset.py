import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import random

def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_label = voice_item['label_id']
    return voice_data, voice_label

def load_face(face_item):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([64, 64])
    face_data = np.transpose(np.array(face_data), (2, 0, 1))
    face_data = ((face_data - 127.5) / 127.5).astype('float32')
    face_label = face_item['label_id']
    return face_data, face_label

class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1]
        self.length = len(self.voice_list)

    def __getitem__(self, index):
        ranidx = random.randint(0, self.length-1)
        voice_data, voice_label = load_voice(self.voice_list[index])
        if index == self.length-1:
            p_ind = index-1
        else:
            p_ind = index+1
        voice_data_p, _ = load_voice(self.voice_list[p_ind])
        voice_data_n, _ = load_voice(self.voice_list[ranidx])
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]
        pt_p = np.random.randint(voice_data_p.shape[1] - self.crop_nframe + 1)
        voice_data_p = voice_data_p[:, pt_p:pt_p+self.crop_nframe]
        pt_n = np.random.randint(voice_data_n.shape[1] - self.crop_nframe + 1)
        voice_data_n = voice_data_n[:, pt_n:pt_n+self.crop_nframe]
        return voice_data, voice_label, voice_data_p, voice_data_n

    def __len__(self):
        return len(self.voice_list)

class FaceDataset(Dataset):
    def __init__(self, face_list):
        self.face_list = face_list

    def __getitem__(self, index):
        face_data, face_label = load_face(self.face_list[index])
        if np.random.random() > 0.5:
           face_data = np.flip(face_data, axis=2).copy()
        return face_data, face_label

    def __len__(self):
        return len(self.face_list)
