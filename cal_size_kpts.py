# This script calculates the point-to-point face size (Keypoint)

import numpy as np
import glob
from statistics import mean

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


kpts = np.load('train.configs/keypoints_sim.npy')
folders = glob.glob('data/all_test_result_3PerP_supervised_64/*')
kpts_name = []

for folder in folders:
    folder_name = folder.rsplit('/',1)[-1]
    print("Evaluating: ", folder_name)
    all_predictions = glob.glob(folder+'/*.obj')
    target_pts = read_xyz(glob.glob('data/A2E_val/'+folder_name+'/*.xyz')[0])
    target_OICD = np.linalg.norm(target_pts[2217]-target_pts[14607])

    RMSE_col = []

    for pred in all_predictions:
        pred_pts = read_obj(pred)
        pred_OICD = np.linalg.norm(pred_pts[2217]-pred_pts[14607])
        pred_pts *= (target_OICD/pred_OICD)
        pred_pts_flat = pred_pts.flatten(order='C')
        target_pts_flat = target_pts.flatten(order='C')

        size_R, size_C = target_pts[:,1].max()-target_pts[:,1].min(), target_pts[:,0].max()-target_pts[:,0].min()
        pred_kpts, target_kpts = pred_pts_flat[kpts], target_pts_flat[kpts]
        RMSE = np.linalg.norm(pred_kpts-target_kpts)/np.sqrt(size_R*size_C)
        RMSE_col.append(RMSE)
    
    kpts_name_mean = mean(RMSE_col)
    kpts_name.append(kpts_name_mean)

print("Keypoints error: ", mean(kpts_name))