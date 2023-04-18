import os
import time
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import scipy.io as sio

from torch.utils.data import DataLoader
from config import DATASET_PARAMETERS, NETWORKS_PARAMETERS
from parse_dataset import get_dataset
from network import get_network, SynergyNet
from utils import Meter, cycle, cycle_4, save_model, read_xyz, voice2face_processed, write_obj_with_colors
from distiller_zoo import PKT

import torch.optim as optim
import glob
import numpy as np
from statistics import mean
import logging
from datetime import datetime

if not os.path.exists(NETWORKS_PARAMETERS['SAVE_DIR']):
    os.makedirs(NETWORKS_PARAMETERS['SAVE_DIR'])
logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(NETWORKS_PARAMETERS['SAVE_DIR']+'/{:%Y-%m-%d-%H-%M-%S}.log'.format(datetime.now()), mode='w'),
            logging.StreamHandler()
        ]
    )
logging.info(f'Save the pth at {NETWORKS_PARAMETERS["SAVE_DIR"]}')

# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset(DATASET_PARAMETERS)
NETWORKS_PARAMETERS['c']['output_channel'] = id_class_num


print('Preparing the datasets...')
voice_dataset = DATASET_PARAMETERS['voice_dataset'](voice_list,
                               DATASET_PARAMETERS['nframe_range'])
face_dataset = DATASET_PARAMETERS['face_dataset'](face_list)

print('Preparing the dataloaders...')
collate_fn = DATASET_PARAMETERS['collate_fn'](DATASET_PARAMETERS['nframe_range'])
collate_fn_4 = DATASET_PARAMETERS['collate_fn_4'](DATASET_PARAMETERS['nframe_range'])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size=DATASET_PARAMETERS['batch_size'],
                          num_workers=DATASET_PARAMETERS['workers_num'],
                          collate_fn=collate_fn_4)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size=DATASET_PARAMETERS['batch_size'],
                         num_workers=DATASET_PARAMETERS['workers_num'])

voice_iterator = iter(cycle_4(voice_loader))
face_iterator = iter(cycle(face_loader))

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Initializing networks...')
e_net, e_optimizer = get_network('e', NETWORKS_PARAMETERS, train=False)
g_net, g_optimizer = get_network('g', NETWORKS_PARAMETERS, train=True)
f_net, f_optimizer = get_network('f', NETWORKS_PARAMETERS, train=True)
d_net, d_optimizer = get_network('d', NETWORKS_PARAMETERS, train=True)
c_net, c_optimizer = get_network('c', NETWORKS_PARAMETERS, train=True)

# for image to 3D part
image3D_pretrained = SynergyNet(pretrained=True).cuda().eval()
image3D = SynergyNet().cuda()
up_layer = torch.nn.Upsample((120,120), mode='bilinear', align_corners=True)
dis_optimizer = optim.Adam(image3D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(list(g_net.parameters())+list(image3D.parameters()), lr=0.0002, betas=(0.5, 0.999))
voice_list = sorted(glob.glob('data/val_sub/*'))
tri = sio.loadmat('./train.configs/tri.mat')['tri']

# distiller zoo- we use PKT here; refer to the zoo for more options.
distiller = PKT()
tripLoss = torch.nn.TripletMarginLoss()

# label for real/fake faces
real_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 1).float()
fake_label = torch.full((DATASET_PARAMETERS['batch_size'], 1), 0).float()

# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_real = Meter('D_real', 'avg', ':3.2f')
D_fake = Meter('D_fake', 'avg', ':3.2f')
C_real = Meter('C_real', 'avg', ':3.2f')
GD_fake = Meter('G_D_fake', 'avg', ':3.2f')
GC_fake = Meter('G_C_fake', 'avg', ':3.2f')
Distill = Meter('Distill', 'avg', ':3.2f')
Trip = Meter('Triplet', 'avg', ':3.2f')

# Validation point set 

print('Training models...')
for it in range(50000):
    # data
    start_time = time.time()
    
    voice, voice_label, voice_p, voice_n = next(voice_iterator)
    face, face_label = next(face_iterator)
    noise = 0.05*torch.randn(DATASET_PARAMETERS['batch_size'], 64, 1, 1)

    # use GPU or not
    if NETWORKS_PARAMETERS['GPU']: 
        voice, voice_label = voice.cuda(), voice_label.cuda()
        face, face_label = face.cuda(), face_label.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()
        voice_p, voice_n = voice_p.cuda(), voice_n.cuda()
    data_time.update(time.time() - start_time)

    # get embeddings and generated faces
    embeddings = e_net(voice)
    embeddings = F.normalize(embeddings)
    # introduce some permutations
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    fake = g_net(embeddings)

    # get embeddings and generated faces
    embeddings_p = e_net(voice_p)
    embeddings_p = F.normalize(embeddings_p)
    # introduce some permutations
    embeddings_p = embeddings_p + noise
    embeddings_p = F.normalize(embeddings_p)
    fake_p = g_net(embeddings_p)

    # get embeddings and generated faces
    embeddings_n = e_net(voice_n)
    embeddings_n = F.normalize(embeddings_n)
    # introduce some permutations
    embeddings_n = embeddings_n + noise
    embeddings_n = F.normalize(embeddings_n)
    fake_n = g_net(embeddings_n)

    # Discriminator
    f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    real_score_out = d_net(f_net(face))
    fake_score_out = d_net(f_net(fake.detach()))
    real_label_out = c_net(f_net(face))
    D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label)
    D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label)
    C_real_loss = F.nll_loss(F.log_softmax(real_label_out, 1), face_label)
    D_real.update(D_real_loss.item())
    D_fake.update(D_fake_loss.item())
    C_real.update(C_real_loss.item())
    (D_real_loss + D_fake_loss + C_real_loss).backward()
    f_optimizer.step()
    d_optimizer.step()
    c_optimizer.step()


    ## Joint training
    g_optimizer.zero_grad()
    fake_score_out = d_net(f_net(fake))
    fake_label_out = c_net(f_net(fake))
    face_image =  up_layer(fake)
    face_image_p =  up_layer(fake_p)
    face_image_n =  up_layer(fake_n)
    prediction_pre, pool_pre, inter_pre = image3D_pretrained(face_image, return_interFeature=True)
    prediction, pool, inter = image3D(face_image, return_interFeature=True)
    prediction_p = image3D(face_image_p)
    prediction_n = image3D(face_image_n)

    GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label)
    GC_fake_loss = F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
    # distillation loss
    distill_loss = 0.5 * F.mse_loss(prediction_pre, prediction) + 10000*(distiller(pool_pre, pool) + distiller(inter_pre.view(inter_pre.shape[0],-1), inter.view(inter.shape[0],-1)))
    # triplet loss
    triplet_loss = 1.5 * tripLoss(prediction, prediction_p, prediction_n)
    (GD_fake_loss + GC_fake_loss + distill_loss + triplet_loss).backward()
    GD_fake.update(GD_fake_loss)
    GC_fake.update(GC_fake_loss.item())
    Distill.update(distill_loss.item())
    Trip.update(triplet_loss.item())
    g_optimizer.step()

    batch_time.update(time.time() - start_time)

    # print status
    if it % 2000 == 0:
        msg = str(iteration)+str(data_time)+str(batch_time)+str(D_real)+str(D_fake)+str(C_real)+str(GD_fake)+str(GC_fake)+str(Distill)+str(Trip)
        
        logging.info(msg)

        data_time.reset()
        batch_time.reset()
        D_real.reset()
        D_fake.reset()
        C_real.reset()
        GD_fake.reset()
        GC_fake.reset()
        Distill.reset()
        Trip.reset()

        e_net.eval()
        g_net.eval()
        image3D.eval()
        fore_err, cheek_err, ear_err, mid_err = [],[],[],[]

        for folder in voice_list:
            name = folder.split('/',1)[-1]
            all_fbanks = glob.glob(folder+'/*.npy')
            target_pts = read_xyz(glob.glob('data/AtoE_sub/'+name+'/*.xyz')[0])

            target_OICD = np.linalg.norm(target_pts[2217]-target_pts[14607])
            target_foreD = np.linalg.norm(target_pts[1678]-target_pts[42117])
            target_cheekD = np.linalg.norm(target_pts[2294]-target_pts[13635])
            target_earD = np.linalg.norm(target_pts[20636]-target_pts[34153])
            target_midD = np.linalg.norm(target_pts[2130]-target_pts[15003])

            target_foreOICD = target_foreD/target_OICD
            target_cheekOICD = target_cheekD/target_OICD
            target_earOICD = target_earD/target_OICD
            target_midOICD = target_midD/target_OICD

            for fbank in all_fbanks:
                face_image = voice2face_processed(e_net, g_net, fbank,NETWORKS_PARAMETERS['GPU'])
                face_image = up_layer(face_image)
                pred_pts = image3D(face_image)[0].squeeze().transpose(1,0).detach().cpu()

                # simple validation
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
        val_msg = f'Val forehead: {fore_err_mean:.4f}, cheek: {cheek_err_mean:.4f}, ear: {ear_err_mean:.4f}, mid: {mid_err_mean:.4f}'
        
        logging.info(val_msg)

        # reset to train
        e_net.train()
        g_net.train()
        image3D.train()

        # snapshot
        save_model(g_net, NETWORKS_PARAMETERS['g']['model_path'][:-4]+'_'+str(it)+'.pth')
        save_model(image3D, NETWORKS_PARAMETERS['image3D']['model_path'][:-4]+'_'+str(it)+'.pth')

    iteration.update(1)

