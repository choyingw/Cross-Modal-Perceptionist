# This script calculates the point-to-point face size (ARE)

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

folders = glob.glob('data/all_test_result_3PerP_supervised_64/*')
fore_name, cheek_name, ear_name, mid_name = [], [], [], []

for folder in folders:
    folder_name = folder.rsplit('/',1)[-1]
    print("Evaluating: ", folder_name)
    all_predictions = glob.glob(folder+'/*.obj')
    target_pts = read_xyz(glob.glob('data/A2E_val/'+folder_name+'/*.xyz')[0])
    target_OICD = np.linalg.norm(target_pts[2217]-target_pts[14607])
    target_foreD = np.linalg.norm(target_pts[1678]-target_pts[42117])
    target_cheekD = np.linalg.norm(target_pts[2294]-target_pts[13635])
    target_earD = np.linalg.norm(target_pts[20636]-target_pts[34153])
    target_midD = np.linalg.norm(target_pts[2130]-target_pts[15003])

    target_foreOICD = target_foreD/target_OICD
    target_cheekOICD = target_cheekD/target_OICD
    target_earOICD = target_earD/target_OICD
    target_midOICD = target_midD/target_OICD

    fore_err, cheek_err, ear_err, mid_err = [],[],[],[]

    for pred in all_predictions:
        pred_pts = read_obj(pred)
        pred_OICD = np.linalg.norm(pred_pts[2217]-pred_pts[14607])
        pred_pts *= (target_OICD/pred_OICD)
        pred_OICD = np.linalg.norm(pred_pts[2217]-pred_pts[14607])
        pred_midD = np.linalg.norm(pred_pts[2130]-pred_pts[15003])
        pred_foreD = np.linalg.norm(pred_pts[1678]-pred_pts[42117])
        pred_cheekD = np.linalg.norm(pred_pts[2294]-pred_pts[13635])
        pred_earD = np.linalg.norm(pred_pts[20636]-pred_pts[34153])

        pred_midOICD = pred_midD/pred_OICD
        pred_foreOICD = pred_foreD/pred_OICD
        pred_cheekOICD = pred_cheekD/pred_OICD
        pred_earOICD = pred_earD/pred_OICD

        fore_err.append(abs(pred_foreOICD-target_foreOICD))
        cheek_err.append(abs(pred_cheekOICD-target_cheekOICD))
        ear_err.append(abs(pred_earOICD-target_earOICD))
        mid_err.append(abs(pred_midOICD-target_midOICD))
    

    fore_err_mean, cheek_err_mean, ear_err_mean, mid_err_mean = mean(fore_err), mean(cheek_err), mean(ear_err), mean(mid_err)
    fore_name.append(fore_err_mean)
    cheek_name.append(cheek_err_mean)
    mid_name.append(mid_err_mean)
    ear_name.append(ear_err_mean)

print("Summary of the ARE:")
print("-----------------------")
print("Fore ratio error", mean(fore_name))
print("Cheek ratio error", mean(cheek_name))
print("Ear ratio error", mean(ear_name))
print("Mid ratio error", mean(mid_name))
