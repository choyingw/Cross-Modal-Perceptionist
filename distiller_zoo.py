import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
	def __init__(self, p=2):
		super(Attention, self).__init__()
		self.p = p


	def forward(self, f_s, f_t):
		if f_s.dim() == 2:
			return (F.normalize(f_s.pow(self.p))-F.normalize(f_t.pow(self.p))).pow(2).mean()
		else:
			return (self.at(f_s) - self.at(f_t)).pow(2).mean()

	def at(self, f):
		return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

class Similarity(nn.Module):
	def __init__(self):
		super(Similarity, self).__init__()


	def forward(self, f_s, f_t):
		bsz = f_s.shape[0]
		f_s = f_s.view(bsz, -1)
		f_t = f_t.view(bsz, -1)
		G_s = torch.mm(f_s, torch.t(f_s))
		G_s = torch.nn.functional.normalize(G_s)

		G_t = torch.mm(f_t, torch.t(f_t))
		G_t = torch.nn.functional.normalize(G_t)

		G_diff = G_t - G_s
		loss = (G_diff*G_diff).view(-1, 1).sum(0)/(bsz*bsz)
		return loss

class Correlation(nn.Module):
	def __init__(self):
		super(Correlation, self).__init__()

	def forward(self, f_s, f_t):
		delta = torch.abs(f_s-f_t)
		loss = torch.mean((delta[:-1]*delta[1:]).sum(1))
		return loss

class NSTLoss(nn.Module):
	def __init__(self):
		super(NSTLoss, self).__init__()
		pass

	def forward(self, f_s, f_t):
		
		if f_s.dim() == 4:
			s_H, t_H = f_s.shape[2], f_t.shape[2]
			if s_H > t_H:
				f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
			elif s_H < t_H:
				f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
			else:
				pass

			f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
			f_s = F.normalize(f_s, dim=2)
			f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
			f_t = F.normalize(f_t, dim=2)

		elif f_s.dim() == 2:
			f_s = F.normalize(f_s, dim=1)
			f_t = F.normalize(f_t, dim=1)

		full_loss = True
		if full_loss:
			return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s,f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean())
		else:
			return self.poly_kernel(f_s, f_s).mean()

	def poly_kernel(self, a, b):
		a = a.unsqueeze(1)
		b = b.unsqueeze(2)
		res = (a*b).sum(-1).pow(2)
		return res

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0
        
        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss

class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match

        # s_H, t_H = input.shape[2], target.shape[2]
        # if s_H > t_H:
        #     input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        # elif s_H < t_H:
        #     target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        # else:
        #     pass
        if input.dim() == 2:
        	input = input.unsqueeze(2).unsqueeze(2)
        	target = target.unsqueeze(2).unsqueeze(2)

        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)

        return loss