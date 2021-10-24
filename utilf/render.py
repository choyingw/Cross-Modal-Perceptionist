import sys

sys.path.append('..')

import cv2
import numpy as np
import scipy.io as sio

from Sim3DR import RenderPipeline

# to continuous
def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

# load BFM connectivity of triangles
tri = sio.loadmat('./train.configs/tri.mat')['tri'] - 1
tri = _to_ctype(tri.T).astype(np.int32)

# Sim3DR definition
cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}

render_app = RenderPipeline(**cfg)

def render_vert(img, vert, alpha=1.0, wfp=None):
    print(f'Save visualization result to {wfp}')
    overlap = img.copy()
    vert = vert.astype(np.float32)
    ver = _to_ctype(vert.T)  # transpose
    overlap = render_app(ver, tri, overlap)
    overlap = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    cv2.imwrite(wfp[:-4]+'.png', overlap)

