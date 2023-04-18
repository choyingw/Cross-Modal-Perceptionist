import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F
import pickle
import os.path as osp

from PIL import Image
from scipy.io import wavfile
from torch.utils.data.dataloader import default_collate
from vad import read_wave, write_wave, frame_generator, vad_collector

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()

def parse_param_102(param):
	"""Work for only tensor"""
	p_ = param[:, :12].reshape(-1, 3, 4)
	p = p_[:, :, :3]
	offset = p_[:, :, -1].reshape(-1, 3, 1)
	alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
	alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
	alpha_tex = param[:, 62:102].reshape(-1, 40, 1)
	return p, offset, alpha_shp, alpha_exp, alpha_tex

def to_rotation_mat_renorm(R):
    s = (R[:, 0, :3].norm(dim=1) + R[:, 1, :3].norm(dim=1))/2.0
    return F.normalize(R, p=2, dim=2), s

class ParamsPack():
    """3DMM configuration data loading from ./train.configs"""
    def __init__(self, version):
        data_ver = version

        d = make_abs_path('./train.configs')

        # PCA basis for shape, expression, texture
        self.w_shp = _load_tensor(osp.join(d, 'w_shp_{}.npy'.format(data_ver)), mode='gpu')
        self.w_exp = _load_tensor(osp.join(d, 'w_exp_{}.npy'.format(data_ver)), mode='gpu')  
        #self.w_tex = torch.from_numpy(_load(osp.join(d, 'w_tex_sim.npy'))[:,:40]).cuda()

        # param_mean and param_std are used for re-whitening
        meta = _load(osp.join(d, 'param_whitening_{}.pkl'.format(data_ver)))
        self.param_mean = torch.from_numpy(meta.get('param_mean')).float().cuda()
        self.param_std = torch.from_numpy(meta.get('param_std')).float().cuda()

        # mean values
        self.u_shp = _load_tensor(osp.join(d, 'u_shp.npy'), mode='gpu')
        self.u_exp = _load_tensor(osp.join(d, 'u_exp.npy'), mode='gpu')
        #self.u_tex = _load_tensor(osp.join(d, 'u_tex.npy'), mode='gpu')
        self.u = self.u_shp + self.u_exp
        self.w = torch.cat((self.w_shp, self.w_exp), dim=1)

        # base vector for landmarks
        self.std_size = 120
        self.dim = self.w_shp.shape[0] // 3

param_pack = ParamsPack('v201')

class Meter(object):
    # Computes and stores the average and current value
    def __init__(self, name, display, fmt=':f'):
        self.name = name
        self.display = display
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{' + self.display  + self.fmt + '},'
        return fmtstr.format(**self.__dict__)

def get_collate_fn(nframe_range):
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1])
                 for item in batch]
        return default_collate(batch)
    return collate_fn

def get_collate_fn_4(nframe_range):
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1], item[2][..., pt:pt+num_frame], item[3][..., pt:pt+num_frame]) for item in batch]
        return default_collate(batch)
    return collate_fn

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

def cycle_4(dataloader):
    while True:
        for data, label, data_p, data_n in dataloader:
            yield data, label, data_p, data_n

def save_model(net, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    torch.save(net.state_dict(), model_path)

def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
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
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # print(fbank.shape)
    # m=fbank.mean(axis=0)
    # print(m.shape)
    # exit()

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


def voice2face_processed(e_net, g_net, fbank_obj, GPU=True, return_embeddings=False):
    fbank = np.load(fbank_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)

    face = g_net(embedding)

    if return_embeddings:
        return face, embedding
    
    return face

def voice2face_processed_ParamOut(e_net, g_net, fbank_obj, GPU=True):
    fbank = np.load(fbank_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net.forward_test(embedding)
    
    return face

def voice2face_processed_MeshOut(e_net, g_net, fbank_obj, GPU=True):
    fbank = np.load(fbank_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net.forward_test(embedding)
    
    return face

def write_obj_with_colors(obj_name, vertices, triangles):
		"""
		write out obj mesh files.
		"""
		if obj_name.split('.')[-1] != 'obj':
			obj_name = obj_name + '.obj'

		# write obj
		with open(obj_name, 'w') as f:
			# write vertices & colors
			for i in range(vertices.shape[1]):
				s = 'v {} {} {}\n'.format(vertices[0, i],  vertices[1, i], vertices[2, i])
				f.write(s)

			# write f: ver ind/ uv ind
			for i in range(triangles.shape[1]):
				s = 'f {} {} {}\n'.format(triangles[0, i], triangles[1, i], triangles[2, i])
				f.write(s)

def read_obj(filename):
    f = open(filename)
    lines = f.readlines()
    coll = []
    for l in lines:
        if l[0] != 'v':
            break
        comp = l.split()[1:]
        comp = list(map(float, comp))
        coll.append(comp)

    a = np.asarray(coll)
    return a

def read_xyz(filename):
    f = open(filename)
    lines = f.readlines()
    coll = []
    for l in lines:
        comp = l.split()
        comp = list(map(float, comp))
        coll.append(comp)
    a=np.asarray(coll)
    return a
