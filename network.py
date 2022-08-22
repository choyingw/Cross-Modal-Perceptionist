import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from backbone_nets import mobilenetv2_backbone

class VoiceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[1], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[2], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm1d(channels[3], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x

class Generator(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class FaceEmbedNet(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[3], channels[4], 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[4], output_channel, 4, 1, 0, bias=True),
        )
 
    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x

def get_network(net_type, params, train=True):
    net_params = params[net_type]
    net = net_params['network'](net_params['input_channel'],
                                net_params['channels'],
                                net_params['output_channel'])
    if params['GPU']:
        net.cuda()

    if train:
        net.train()
        optimizer = optim.Adam(net.parameters(),
                               lr=params['lr'],
                               betas=(params['beta1'], params['beta2']))
    else:
        net.eval()
        net.load_state_dict(torch.load(net_params['model_path']))
        optimizer = None
    return net, optimizer

# SynergyNet module definition
class SynergyNet(nn.Module):
    '''Defintion of 2D-to-3D-part'''
    def __init__(self, pretrained=False, last_CN=None):
        super(SynergyNet, self).__init__()
        self.backbone = getattr(mobilenetv2_backbone, 'mobilenet_v2')(last_CN=last_CN)

        # load the pretained model for 2D-to-3D
        ckpt = torch.load('pretrained_models/2D-to-3D-pretrained.tar')['state_dict']
        model_dict = self.backbone.state_dict()
        for k,v in ckpt.items():
            if 'IGM' in k:
                name_reduced = k.split('.',3)[-1]
                model_dict[name_reduced] = v

        if pretrained: # SynergyNet pretrain
            self.backbone.load_state_dict(model_dict)
        
        # 3DMM parameters and whitening parameters
        self.param_std = ckpt['module.param_std']
        self.param_mean = ckpt['module.param_mean']
        self.w_shp = ckpt['module.w_shp']
        self.w_exp = ckpt['module.w_exp']
        self.u = ckpt['module.u'].unsqueeze(0)

    def forward(self, input, return_onlypose=False):
        _3D_attr = self.backbone(input)
        if return_onlypose:
            # only return pose 
            return _3D_attr[:,:12] * self.param_std[:12] + self.param_mean[:12]
        else:
            # return dense mesh face
            _3D_face = self.reconstruct_vertex(_3D_attr, dense=True)
            return _3D_face

    def reconstruct_vertex(self, param, whitening=True, dense=False):
        '''
        Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks.
        Working with batched tensors. Using Fortan-type reshape.
        '''
        # 12 transformation + 40 shape + 10 expr + 40 (discarded) texture
        if whitening:
            if param.shape[1] == 102:
                param_ = param * self.param_std + self.param_mean
            else:
                raise RuntimeError('length of params mismatch')
        p, _, alpha_shp, alpha_exp = self.parse_param_102(param_)
        _, s = self.p_to_Rs(p)

        # frontal mesh construction with 53215 vertics (BFM Face)
        if dense:
            vertex = s.unsqueeze(1).unsqueeze(1)*(self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).squeeze().contiguous().view(-1, 53215, 3).transpose(1,2)
        else:
            raise NotImplementedError("Only dense mesh reconstruction supported")
        
        return vertex

    def parse_param_102(self, param):
        ''' Parse param into 3DMM semantics'''
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        return p, offset, alpha_shp, alpha_exp

    def parse_param_102_pose(self, param):
        ''' Parse only pose params'''
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        R, s = self.p_to_Rs(p)
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        return R, offset

    def p_to_Rs(self, R):
        '''Convert P to R and s as in 3DDFA-V2'''
        s = (R[:, 0, :3].norm(dim=1) + R[:, 1, :3].norm(dim=1))/2.0
        return F.normalize(R, p=2, dim=2), s
