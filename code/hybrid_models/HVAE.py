#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import math

import torch.nn as nn
from matplotlib import pyplot as plt

from code.simulators import PhysicalModel, NoSimulator
import torch
from code.nn import MLP, act_dict, ConditionalUNet, ConditionalUNetReactionDiffusion, Permute, kl_gaussians, UNet
from torchdiffeq import odeint as odeint
from code.hybrid_models.HybridAutoencoder import HybridAutoencoder


class DynamicalPhysicalDecoder(nn.Module):
    def __init__(self, fp: PhysicalModel):
        super(DynamicalPhysicalDecoder, self).__init__()
        self.fp = fp
        self.z_p = None

    def to(self, device):
        super(DynamicalPhysicalDecoder, self).to(device)
        self.fp.to(device)
        return self

    def set_latent(self, z_p):
        # h: B_size x hidden_size
        self.z_p = z_p

    def get_physical_parameters(self, x):
        fp_params = self.z_p
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)

        return physical_params

    def forward_fp(self, t, x):
        physical_params = self.get_physical_parameters(x)
        return self.fp.parameterized_forward(x, **physical_params)

    def forward(self, t, x):
        dx = self.forward_fp(t, x)
        return dx.unsqueeze(1)


class AugmentedHybridDecoder(nn.Module):
    def __init__(self, fp: PhysicalModel, fa: nn.Module):
        super(AugmentedHybridDecoder, self).__init__()
        # Create net that maps hidden from encoder + state x to fa(x;h) contribution.
        self.fa = fa

        self.fp = fp

        self.x_dim = fp._X_dim

        self.z_a = None
        self.z_p = None
        self.fp_params = None

    def to(self, device):
        super(AugmentedHybridDecoder, self).to(device)
        self.fp.to(device)
        self.fa.to(device)
        return self

    def set_latent(self, z_a, z_p):
        # h: B_size x hidden_size
        self.z_a = z_a
        self.z_p = z_p

    def forward_fa(self, t, x):
        if len(x.shape) == 4:
            x_fa = torch.cat((x, self.z_a.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])), 1)
            return self.fa(x_fa)
            return self.fa(x, self.z_a)
        return self.fa(torch.cat((x, self.z_a), 1))

    def get_physical_parameters(self, x):
        fp_params = self.z_p
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)

        return physical_params

    def forward_fp(self, t, x):
        physical_params = self.get_physical_parameters(x)
        return self.fp.parameterized_forward(t, x, **physical_params)

    def forward(self, t, x):
        if len(x.shape) == 4:
            x, x_r = torch.chunk(x, 2, 1)
            dx, dx_r = torch.chunk(self.forward_fp(t, torch.cat((x, x_r), 0)), 2, 0)
            dx = torch.cat((self.forward_fa(t, x) + dx, dx_r), 2)
            return dx
        x, x_r = torch.chunk(x, 2, 1)
        dx, dx_r = torch.chunk(self.forward_fp(t, torch.cat((x, x_r), 0)), 2, 0)
        dx = torch.cat((self.forward_fa(t, x) + dx, dx_r), 1)
        return dx#.unsqueeze(1)


class HybridVAE(HybridAutoencoder):
    def predicted_parameters_as_dict(self, t_span, x, zero_param=False) -> dict:
        x_obs = x[:, :self.nb_observed].view(-1, self.nb_observed * self.x_dim)

        mu_a, log_sigma_a = torch.chunk(self.ga(x_obs), 2, 1)
        z_a = mu_a

        x_p = x_obs + self.gp_1(torch.cat((x_obs, z_a), 1))
        mu_p, log_sigma_p = torch.chunk(self.gp_2(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)

        fp_params = mu_p
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)
        for i, p in enumerate(self.fp.missing_param_dim_textual):
            physical_params[p] = torch.zeros(x.shape[0], device=x.device).unsqueeze(1)
        return physical_params

    def __init__(self, fp: PhysicalModel, device="cpu", **config):
        super(HybridVAE, self).__init__()
        self.device = device
        self.z_a_dim = 1 if "z_a_dim" not in config else config["z_a_dim"]
        self.nb_observed = 50 if "nb_observed" not in config else config["nb_observed"]
        self.x_dim, self.z_p_dim = fp._X_dim, len(fp.incomplete_param_dim_textual)
        factor = 1
        self.alpha = .01 * factor if "alpha" not in config else config["alpha"]
        self.beta = .001 * factor if "beta" not in config else config["beta"]
        self.gamma = .1 * factor if "gamma" not in config else config["gamma"]
        self.posterior_type = "dirac" if "posterior_type" not in config else config["posterior_type"]

        seq_dim = self.x_dim * self.nb_observed

        mu_prior_zp = []
        sigma_prior_zp = []
        min_zp = []
        max_zp = []
        for k, v in config["zp_priors"].items():
            mu_prior_zp.append(v["mu"])
            sigma_prior_zp.append(v["sigma"])
            min_zp.append(v["min"])
            max_zp.append(v["max"])
        self.mu_prior_zp, self.sigma_prior_zp = torch.tensor(mu_prior_zp, device=device), torch.tensor(sigma_prior_zp,
                                                                                                       device=device)
        self.min_zp, self.max_zp = torch.tensor(min_zp, device=device), torch.tensor(max_zp, device=device)

        # Fp encoders
        gp_1_hidden = [128, 128] if "gp_1_hidden" not in config else config["gp_1_hidden"]
        gp_2_hidden = [128, 128, 256, 64, 32] if "gp_2_hidden" not in config else config["gp_2_hidden"]
        self.gp_1 = MLP([seq_dim + self.z_a_dim] + gp_1_hidden + [seq_dim], 0, "SELU", None).to(device)
        self.gp_2 = MLP([seq_dim] + gp_2_hidden + [self.z_p_dim * 2], 0, "SELU", None).to(device)
        self.act_mu_p = nn.Identity() if "act_mu_p" not in config else act_dict[config["act_mu_p"]]

        # Fa Encoder
        ga_hidden = [256, 256, 128, 32] if "ga_hidden" not in config else config["ga_hidden"]
        self.ga = MLP([seq_dim] + ga_hidden + [self.z_a_dim * 2], 0, "SELU", None).to(device)

        # Hybrid decoder
        fa_hidden = [64, 64] if "fa_hidden" not in config else config["fa_hidden"]
        param_decoder = {"fp": fp.to(device),
                         "layers_fa": [self.z_a_dim + self.x_dim] + fa_hidden + [self.x_dim],
                         "fa_hidden_act": "SELU",
                         "fa_final_act": None}
        fa = MLP(param_decoder["layers_fa"], hidden_act=param_decoder["fa_hidden_act"],
                 final_act=param_decoder["fa_final_act"])
        self.dec = AugmentedHybridDecoder(fp=fp, fa=fa).to(device)
        self.sigma_x = nn.Parameter(torch.zeros(self.x_dim, requires_grad=True)).to(device)

        # Fp only decoder
        self.fp = fp
        self.fp_only = DynamicalPhysicalDecoder(fp)

    def augmented_data(self, t_span, x):
        resampled_zp = torch.rand(x.shape[0], self.z_p_dim, device=self.device) * (self.max_zp - self.min_zp) \
                       + self.min_zp
        resampled_za = torch.randn(x.shape[0], self.z_a_dim, device=self.device)
        self.dec.set_latent(resampled_za, resampled_zp.repeat(2, 1))

        mu_x_pred_all = odeint(self.dec, x[:, 0, :, 0, 0].repeat(1, 2), t_span).permute(1, 0, 2)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 2)

        return mu_x_pred_tot.unsqueeze(3).unsqueeze(3), resampled_zp

    def forward(self, t_span, x):
        x_obs = x[:, :self.nb_observed].view(-1, self.nb_observed * self.x_dim)

        mu_a, log_sigma_a = torch.chunk(self.ga(x_obs), 2, 1)
        z_a = mu_a

        x_p = x_obs + self.gp_1(torch.cat((x_obs, z_a), 1))
        mu_p, log_sigma_p = torch.chunk(self.gp_2(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)
        z_p = mu_p

        self.dec.set_latent(z_a, z_p.repeat(2, 1))
        mu_x_pred_all = odeint(self.dec, x[:, 0, :, 0, 0].repeat(1, 2), t_span).permute(1, 0, 2)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 2)
        return t_span, mu_x_pred_tot.unsqueeze(3).unsqueeze(3)

    def predicted_parameters(self, t_span, x):
        x_obs = x[:, :self.nb_observed].view(-1, self.nb_observed * self.x_dim)

        mu_a, log_sigma_a = torch.chunk(self.ga(x_obs), 2, 1)
        z_a = mu_a

        x_p = x_obs + self.gp_1(torch.cat((x_obs, z_a), 1))
        mu_p, log_sigma_p = torch.chunk(self.gp_2(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)

        fp_params = mu_p
        return fp_params

    def loss(self, t_span, x):
        return self.loss_augm(t_span, x, None)

    def loss_augm(self, t_span, x, zp):
        b_size = x.shape[0]
        x = x[:, :, :, 0, 0]

        x_obs = x[:, :self.nb_observed].view(-1, self.nb_observed * self.x_dim)

        mu_a, log_sigma_a = torch.chunk(self.ga(x_obs), 2, 1)
        sigma_a = torch.exp(log_sigma_a)
        z_a = mu_a + sigma_a * torch.randn_like(mu_a)

        x_p = x_obs + self.gp_1(torch.cat((x_obs, z_a), 1))
        mu_p, log_sigma_p = torch.chunk(self.gp_2(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)
        sigma_p = torch.exp(log_sigma_p)
        ll_zp_augm = torch.distributions.Normal(loc=mu_p, scale=sigma_p).log_prob(zp).sum(1) if zp is not None else 0.

        if self.posterior_type == "positive_gaussian":
            z_p = mu_p + sigma_p * torch.randn_like(mu_p)
            z_p[z_p <= 0.] = 1.
        elif self.posterior_type == "dirac":
            z_p = mu_p
        else:
            raise Exception("The posterior type: %s is not implemented" % self.posterior_type)

        self.dec.set_latent(z_a, z_p.repeat(2, 1))
        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 2), t_span[:self.nb_observed], method='dopri5',
                               atol=1e-5, rtol=1e-5).permute(1, 0, 2)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 2)
        sigma_x_pred = torch.exp(self.sigma_x.unsqueeze(0).unsqueeze(1).repeat(mu_x_pred_tot.shape[0],
                                                                               mu_x_pred_tot.shape[1], 1))
        x_obs = x[:, :self.nb_observed]

        mse_traj = (x_obs - mu_x_pred_tot[:, :self.nb_observed]).norm(2, dim=2).mean(0).mean()
        #print(torch.exp(self.sigma_x))
        #exit()
        ll = torch.distributions.Normal(loc=mu_x_pred_tot, scale=sigma_x_pred).log_prob(x_obs).sum(1).sum(1)

        ELBO = ll - kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a)) \
               - kl_gaussians(mu_p, sigma_p, self.mu_prior_zp.unsqueeze(0).repeat(b_size, 1),
                              self.sigma_prior_zp.unsqueeze(0).repeat(b_size, 1)) + ll_zp_augm

        bound_kl_physics_reg = kl_gaussians(mu_x_pred_tot, sigma_x_pred, mu_x_pred_fp, sigma_x_pred).sum(1) \
                               + kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a)) \
                               + kl_gaussians(mu_p, sigma_p, torch.ones_like(mu_p) * self.mu_prior_zp,
                                              torch.ones_like(sigma_p) * self.sigma_prior_zp)

        x_r_detached = mu_x_pred_fp.detach().requires_grad_(True)
        x_p = x_p.view(b_size, self.nb_observed, self.x_dim)
        R_da_1 = ((x_p - x_r_detached) ** 2).sum(1).sum(1)

        resampled_zp = torch.rand(x.shape[0], self.z_p_dim, device=self.device) * (self.max_zp - self.min_zp) \
                       + self.min_zp
        resampled_za = torch.randn(x.shape[0], self.z_a_dim, device=self.device)
        # print(resampled_zp.shape)
        self.dec.set_latent(resampled_za, resampled_zp.repeat(2, 1))
        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 2), t_span[:self.nb_observed], method='dopri5',
                               atol=1e-5, rtol=1e-5)

        mu_x_pred_all = mu_x_pred_all.squeeze(2).permute(1, 0, 2)
        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 2)

        x_r_detached = mu_x_pred_fp.detach().squeeze(2).permute(1, 0, 2).requires_grad_(True)
        mu_p, log_sigma_p = torch.chunk(self.gp_2(x_r_detached.contiguous().view(b_size, -1)), 2, 1)
        R_da_2 = ((mu_p - resampled_zp) ** 2).sum(1)

        loss_tot = -ELBO + self.alpha * bound_kl_physics_reg + self.beta * R_da_1 + self.gamma * R_da_2
        return loss_tot.mean(), torch.tensor(-1.), mse_traj.detach()


class HybridVAEReactionDiffusion(nn.Module):
    def __init__(self, fp: PhysicalModel, device="cpu", **config):
        super(HybridVAEReactionDiffusion, self).__init__()
        self.device = device
        self.z_a_dim = 10 if "z_a_dim" not in config else config["z_a_dim"]
        self.nb_observed = 10 if "nb_observed" not in config else config["nb_observed"]
        self.x_dim, self.z_p_dim = fp._X_dim, len(fp.incomplete_param_dim_textual)
        factor = 1
        self.alpha = .01 * factor if "alpha" not in config else config["alpha"]
        self.beta = .001 * factor if "beta" not in config else config["beta"]
        self.gamma = .1 * factor if "gamma" not in config else config["gamma"]

        mu_prior_zp = []
        sigma_prior_zp = []
        min_zp = []
        max_zp = []
        for k, v in config["zp_priors"].items():
            mu_prior_zp.append(v["mu"])
            sigma_prior_zp.append(v["sigma"])
            min_zp.append(v["min"])
            max_zp.append(v["max"])
        self.mu_prior_zp, self.sigma_prior_zp = torch.tensor(mu_prior_zp, device=device), torch.tensor(sigma_prior_zp,
                                                                                                       device=device)
        self.min_zp, self.max_zp = torch.tensor(min_zp, device=device), torch.tensor(max_zp, device=device)

        # Fp encoders
        self.gp_1 = ConditionalUNetReactionDiffusion(z_a_dim=self.z_a_dim).to(device)
        self.act_mu_p = act_dict["ReactionDiffusionParametersScaler"]

        # Hybrid decoder
        fa = nn.Sequential(nn.Conv2d(2 + self.z_a_dim, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 2, 3, padding=1))
        self.dec = AugmentedHybridDecoder(fp=fp, fa=fa).to(device)
        self.sigma_x = torch.zeros(self.x_dim, requires_grad=False).to(device)  # - 3.

        # Fp only decoder
        self.fp = fp
        self.fp_only = DynamicalPhysicalDecoder(fp)

        self.enc_common = nn.Sequential(nn.Flatten(0, 1), nn.Conv2d(2, 16, 3), nn.ReLU(),
                                        nn.Conv2d(16, 32, 3), nn.AvgPool2d(2),
                                        nn.Conv2d(32, 64, 3), nn.ReLU(),
                                        nn.Conv2d(64, 64, 3), nn.AvgPool2d(2),
                                        nn.Conv2d(64, 32, 3), nn.ReLU(), nn.Unflatten(0, (-1, self.nb_observed)),
                                        Permute((0, 2, 1, 3, 4)), nn.Conv3d(32, 16, 2), nn.ReLU(),
                                        nn.Conv3d(16, 16, 2), nn.Flatten(1, 4))

        self.enc_za = nn.Sequential(self.enc_common,
                                    nn.Linear(128, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 2 * self.z_a_dim))

        self.enc_zp = nn.Sequential(self.enc_common,
                                    nn.Linear(128, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 2 * self.z_p_dim))

    def to(self, device):
        super(HybridVAEReactionDiffusion, self).to(device)
        self.sigma_x = self.sigma_x.to(device)
        self.device = device
        self.min_zp, self.max_zp = self.min_zp.to(device), self.max_zp.to(device)
        return self

    def augmented_data(self, t_span, x):
        b_size = x.shape[0]
        x_obs = x[:, :self.nb_observed]

        resampled_zp = torch.rand(b_size, self.z_p_dim, device=self.device) * (self.max_zp - self.min_zp) + self.min_zp
        resampled_za = torch.randn(b_size, self.z_a_dim, device=self.device)
        self.dec.set_latent(resampled_za, resampled_zp.repeat(2, 1))

        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 1, 2, 1), t_span, method='dopri5', atol=1e-5, rtol=1e-5)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 3)
        mu_x_pred_tot = mu_x_pred_tot.permute(1, 0, 2, 3, 4).reshape(b_size, -1)
        return mu_x_pred_tot, resampled_zp

    def forward(self, t_span, x):
        b_size, im_size = x.shape[0], x.shape[-1]
        x_obs = x[:, :self.nb_observed]

        mu_a, log_sigma_a = torch.chunk(self.enc_za(x_obs), 2, 1)
        z_a = mu_a

        x_p = x_obs + self.gp_1(x_obs.reshape(-1, 2, im_size, im_size),
                                z_a.unsqueeze(1).expand(-1, self.nb_observed, -1).reshape(-1, self.z_a_dim)).reshape(
            b_size, self.nb_observed, 2, im_size, im_size)

        mu_p, log_sigma_p = torch.chunk(self.enc_zp(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)
        z_p = mu_p
        self.dec.set_latent(z_a, z_p.repeat(2, 1))

        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 1, 2, 1), t_span, method='dopri5',
                               atol=1e-5, rtol=1e-5)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 3)
        mu_x_pred_tot = mu_x_pred_tot.permute(1, 0, 2, 3, 4).reshape(b_size, -1)
        mu_x_pred_fp = mu_x_pred_fp.permute(1, 0, 2, 3, 4).reshape(b_size, -1)

        return t_span, mu_x_pred_tot, mu_x_pred_fp

    def predicted_parameters(self, t_span, x):
        b_size, im_size = x.shape[0], x.shape[-1]
        x_obs = x[:, :self.nb_observed]

        mu_a, log_sigma_a = torch.chunk(self.enc_za(x_obs), 2, 1)
        z_a = mu_a

        x_p = x_obs + self.gp_1(x_obs.reshape(-1, 2, im_size, im_size),
                                z_a.unsqueeze(1).expand(-1, self.nb_observed, -1).reshape(-1, self.z_a_dim)).reshape(
            b_size, self.nb_observed, 2, im_size, im_size)

        mu_p, log_sigma_p = torch.chunk(self.enc_zp(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)

        fp_params = mu_p
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)
        for i, p in enumerate(self.fp.missing_param_dim_textual):
            physical_params[p] = torch.zeros(x.shape[0], device=x.device).unsqueeze(1)
        return physical_params

    def loss_augm(self, t_span, x, zp):
        return self.loss(t_span, x, zp)

    def loss(self, t_span, x, zp=None):
        b_size, im_size = x.shape[0], x.shape[-1]

        x_obs = x[:, :self.nb_observed]

        mu_a, log_sigma_a = torch.chunk(self.enc_za(x_obs), 2, 1)

        sigma_a = torch.exp(log_sigma_a)
        z_a = mu_a + sigma_a * torch.randn_like(mu_a)

        delta = self.gp_1(x_obs.reshape(-1, 2, im_size, im_size),
                          z_a.unsqueeze(1).expand(-1, self.nb_observed, -1).reshape(-1, self.z_a_dim)).reshape(b_size,
                                                                                                               self.nb_observed,
                                                                                                               2, im_size,
                                                                                                               im_size)
        x_p = x_obs + delta
        mu_p, log_sigma_p = torch.chunk(self.enc_zp(x_p), 2, 1)
        mu_p = self.act_mu_p(mu_p)
        sigma_p = torch.exp(log_sigma_p)

        ll_zp_augm = torch.distributions.Normal(loc=mu_p, scale=sigma_p).log_prob(zp).sum(1) if zp is not None else 0.

        z_p = mu_p
        self.dec.set_latent(z_a, z_p.repeat(2, 1))
        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 1, 2, 1), t_span[:self.nb_observed], method='dopri5',
                               atol=1e-5, rtol=1e-5)

        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 3)
        mu_x_pred_tot = mu_x_pred_tot.permute(1, 0, 2, 3, 4).reshape(b_size, -1)
        mu_x_pred_fp = mu_x_pred_fp.permute(1, 0, 2, 3, 4).reshape(b_size, -1)

        sigma_x_pred = torch.exp(
            self.sigma_x.unsqueeze(0).unsqueeze(0).repeat(b_size, self.nb_observed, 1, 1, 1).reshape(b_size, -1))

        x_obs = x[:, :self.nb_observed]
        ll = torch.distributions.Normal(loc=mu_x_pred_tot, scale=sigma_x_pred).log_prob(x_obs.view(b_size, -1)).sum(1)
        ELBO = ll - kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a)) + ll_zp_augm\
               - kl_gaussians(mu_p, sigma_p, self.mu_prior_zp.unsqueeze(0).repeat(b_size, 1),
                              self.sigma_prior_zp.unsqueeze(0).repeat(b_size, 1))

        bound_kl_physics_reg = kl_gaussians(mu_x_pred_tot, sigma_x_pred,
                                            mu_x_pred_fp, sigma_x_pred) \
                               + kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a)) \
             + kl_gaussians(mu_p, sigma_p, torch.ones_like(mu_p) * self.mu_prior_zp,
                       torch.ones_like(sigma_p) * self.sigma_prior_zp)

        x_r_detached = mu_x_pred_fp.detach().requires_grad_(True)
        x_p = x_p.view(b_size, -1)
        R_da_1 = ((x_p - x_r_detached) ** 2).sum(1)

        resampled_zp = torch.rand(b_size, self.z_p_dim, device=self.device) * (self.max_zp - self.min_zp) \
                       + self.min_zp
        resampled_za = torch.randn(b_size, self.z_a_dim, device=self.device)
        self.dec.set_latent(resampled_za, resampled_zp.repeat(2, 1))
        mu_x_pred_all = odeint(self.dec, x[:, 0].repeat(1, 1, 2, 1), t_span[:self.nb_observed], method='dopri5',
                               atol=1e-5, rtol=1e-5)

        mu_x_pred_all = mu_x_pred_all
        mu_x_pred_tot, mu_x_pred_fp = torch.chunk(mu_x_pred_all, 2, 3)
        x_r_detached = mu_x_pred_fp.detach().requires_grad_(True)
        mu_p, log_sigma_p = torch.chunk(self.enc_zp(x_r_detached), 2, 1)

        R_da_2 = ((mu_p - resampled_zp) ** 2).sum(1)

        loss_tot = -ELBO + self.alpha * bound_kl_physics_reg + self.beta * R_da_1 + self.gamma * R_da_2
        return loss_tot


class DoublePendulumEncoder(nn.Module):
    def __init__(self, layers=[300, 300, 300], za_dim=50, ze_dim=4, initial_guess=True,
                 nb_observed_theta_0=25, nb_observed_theta_1=25, obtain_init_position=False,
                 **config):
        super(DoublePendulumEncoder, self).__init__()
        self.za_dim = za_dim
        self.ze_dim = ze_dim
        in_dim = 2 # cos and sin
        self.obtain_init_position = obtain_init_position
        self.initial_guess = initial_guess
        self.total_time = nb_observed_theta_0 + nb_observed_theta_1
        self.nb_observed_theta_0 = nb_observed_theta_0
        self.nb_observed_theta_1 = nb_observed_theta_1
        self.simple_encoder = config.get("simple_encoder", False)

        lis = [self.total_time*in_dim] + layers
        if self.simple_encoder:
            los = layers + [za_dim * 2 + ze_dim*2]
        else:
            los = layers + [za_dim*2]
        layers_nn = []
        for li, lo in zip(lis, los):
            layers_nn += [nn.Linear(li, lo), nn.SELU()]
        layers_nn.pop()
        self.enc_za = nn.Sequential(*layers_nn)

        lis = [self.total_time * in_dim + za_dim] + layers
        los = layers + [self.total_time * in_dim]
        layers_nn = []
        for li, lo in zip(lis, los):
            layers_nn += [nn.Linear(li, lo), nn.SELU()]
        layers_nn.pop()
        if self.simple_encoder:
            self.clean_x = lambda x: x
        else:
            self.clean_x = nn.Sequential(*layers_nn)

        if not self.simple_encoder:
            lis = [self.total_time*in_dim] + layers
            los = layers + [ze_dim*2]
            layers_nn = []
            for li, lo in zip(lis, los):
                layers_nn += [nn.Linear(li, lo), nn.SELU()]
            layers_nn.pop()
            self.enc_ze = nn.Sequential(*layers_nn)


    def forward(self, t, x):
        in_s = t.shape[0]
        frequency = 1 / t[1]
        b_size = x.shape[0]
        x_masked = torch.cat((x[:, -self.nb_observed_theta_0:, 0], x[:, -self.nb_observed_theta_1:, 1]), 1).reshape(b_size, -1)

        sin_cos_encoded = torch.cat((torch.sin(x_masked), torch.cos(x_masked)), 1)
        q_z_a = self.enc_za(sin_cos_encoded)
        if self.simple_encoder:
            q_z_e = q_z_a[:, :2*self.ze_dim]
            q_z_a = q_z_a[:, 2 * self.ze_dim:]
            mu_z_a, log_sigma_z_a = torch.chunk(q_z_a, 2, 1)
            mu_z_e, log_sigma_z_e = torch.chunk(q_z_e, 2, 1)
            z_e = mu_z_e + torch.randn_like(mu_z_e) * torch.exp(log_sigma_z_e)
            if self.obtain_init_position:
                z_e = torch.cat([x[:, 0, :2, 0, 0], z_e[:, 2:]], 1)
            z_a = mu_z_a + torch.randn_like(mu_z_a) * torch.exp(log_sigma_z_a)
            return z_e, z_a, q_z_e, q_z_a, sin_cos_encoded

        mu_z_a, log_sigma_z_a = torch.chunk(q_z_a, 2, 1)
        z_a = mu_z_a + torch.randn_like(mu_z_a) * torch.exp(log_sigma_z_a)

        x_clean = sin_cos_encoded + self.clean_x(torch.cat([sin_cos_encoded, z_a], 1))

        q_z_e = self.enc_ze(x_clean)
        mu_z_e, log_sigma_z_e = torch.chunk(q_z_e, 2, 1)
        z_e = mu_z_e + torch.randn_like(mu_z_e) * torch.exp(log_sigma_z_e)

        if self.obtain_init_position:
            z_e = torch.cat([x[:, 0, :2, 0, 0], z_e[:, 2:]], 1)

        z_a = None if self.za_dim == 0 else z_a
        return z_e, z_a, q_z_e, q_z_a, x_clean


class HybridVAEDoublePendulum(HybridAutoencoder):
    def __init__(self, fp: PhysicalModel, device="cpu", **config):
        super(HybridVAEDoublePendulum, self).__init__()
        self.device = device
        self.nb_observed = config.get("nb_observed", 25)
        self.nb_observed_theta_0 = config.get("nb_observed_theta_0", 25)
        self.nb_observed_theta_1 = config.get("nb_observed_theta_1", 25)
        self.za_dim = config.get("za_dim", 5)
        self.zp_dim = 4
        factor = 1
        self.alpha = config.get("alpha", .01 * factor)
        self.beta = config.get("beta", .001 * factor)
        self.gamma = config.get("gamma", .1 * factor)
        self.posterior_type = config.get("posterior_type", "dirac")
        self.zp_prior_type = config.get("zp_prior_type", "Normal")
        self.obtain_init_position = config.get("obtain_init_position", False)
        self.use_complete_signal = config.get("use_complete_signal", False)
        self.no_fa = config.get("no_fa", False)
        self.no_fp = config.get("no_fp", False)

        seq_dim_cos_sin = 2 * (self.nb_observed_theta_0 + self.nb_observed_theta_1)
        self.x_dim = 4 * self.nb_observed

        mu_prior_zp = []
        sigma_prior_zp = []
        min_zp = []
        max_zp = []
        for k, v in config["zp_priors"].items():
            if self.zp_prior_type == "Normal":
                mu_prior_zp.append(v["mu"])
                sigma_prior_zp.append(v["sigma"])
            min_zp.append(v["min"])
            max_zp.append(v["max"])
        if self.zp_prior_type == "Normal":
            self.mu_prior_zp, self.sigma_prior_zp = torch.tensor(mu_prior_zp, device=device), \
                                                    torch.tensor(sigma_prior_zp, device=device)
        self.min_zp, self.max_zp = torch.tensor(min_zp, device=device), torch.tensor(max_zp, device=device)

        # Fp encoders
        gp_1_hidden = [200, 200, 200] if "gp_1_hidden" not in config else config["gp_1_hidden"]
        gp_2_hidden = [200, 200, 200] if "gp_2_hidden" not in config else config["gp_2_hidden"]
        # We use the cardinal coordinates rather than polar ones.
        self.gp_1 = MLP([seq_dim_cos_sin + self.za_dim] + gp_1_hidden + [seq_dim_cos_sin], 0, "SELU", None).to(device)
        self.gp_2 = MLP([seq_dim_cos_sin] + gp_2_hidden + [self.zp_dim * 2], 0, "SELU", None).to(device)

        def act_double_pendulum_parameters(mu_p):
            return torch.cat([math.pi*(2*torch.sigmoid(mu_p[:, :2]) - 1.), mu_p[:, 2:]], 1)

        self.act_mu_p = nn.Identity() #if "act_mu_p" not in config else act_dict[config["act_mu_p"]]

        # Fa Encoder
        ga_hidden = [300] * 3 if "ga_hidden" not in config else config["ga_hidden"]
        self.ga = MLP([seq_dim_cos_sin] + ga_hidden + [self.za_dim * 2], 0, "SELU", None).to(device)
        self.enc = DoublePendulumEncoder(layers=ga_hidden, ze_dim=self.zp_dim,
                                         **config).to(device)

        # Hybrid decoder
        fa_hidden = 3 * [300] if "fa_hidden" not in config else config["fa_hidden"]
        param_decoder = {"fp": fp.to(device),
                         "layers_fa": [self.za_dim + 6] + fa_hidden + [4],
                         "fa_hidden_act": "SELU",
                         "fa_final_act": None}
        fa = MLP(param_decoder["layers_fa"], hidden_act=param_decoder["fa_hidden_act"],
                 final_act=param_decoder["fa_final_act"])
        self.fp = NoSimulator() if self.no_fp else fp
        self.fa = fa
        self.sigma_x_cos = nn.Parameter(torch.zeros(2, requires_grad=True)).to(device)
        self.sigma_x_sin = nn.Parameter(torch.zeros(2, requires_grad=True)).to(device)

        if True:
            self.param_ode_solver = {"method": "dopri5",
                                     "rtol": 1e-8,
                                     "atol": 1e-8
                                     }
        else:
            self.param_ode_solver = {"method": "rk4",
                                     "rtol": 1e-5,
                                     "atol": 1e-5,
                                     "options": {"step_size": .0001}
                                     }
        # Fp only decoder
        self.fp = NoSimulator() if self.no_fp else fp
        self.fp_only = DynamicalPhysicalDecoder(fp)

    def ode_f(self, t, theta, z_a):
        theta_fp_fa, theta_fp_only = torch.chunk(theta, 2, 1)
        ode_fp_only = self.fp(t, theta_fp_only)
        if self.za_dim > 0 and not self.no_fa:
            ode_fp_fa = self.fp(t, theta_fp_fa) + self.fa(torch.cat((torch.sin(theta_fp_fa[:, :2]),
                                                                     torch.cos(theta_fp_fa[:, :2]),
                                                                     theta_fp_fa[:, 2:], z_a), 1))

        else:
            ode_fp_fa = self.fp(t, theta_fp_fa)

        return torch.cat((ode_fp_fa, ode_fp_only), 1)

    def augmented_data(self, t_span, x):   
        with torch.no_grad():
            theta_0, z_a, q_z_e, q_z_a, x_clean = self.enc(t_span, x[:, :self.nb_observed])

            mu_a, log_sigma_a = torch.chunk(q_z_a, 2, 1)
            z_a = mu_a#torch.randn_like(mu_a)
            resampled_theta_0 = torch.rand(x.shape[0], self.min_zp.shape[0], device=self.device) * \
                                (self.max_zp - self.min_zp) + self.min_zp
            
            dec = lambda t, theta: self.fp(t, theta) + self.fa(torch.cat((torch.sin(theta[:, :2]),
                                                                                torch.cos(theta[:, :2]),
                                                                                theta[:, 2:], z_a), 1))
            x_pred = odeint(dec, resampled_theta_0, t_span, **self.param_ode_solver).permute(1, 0, 2)[:, :, :2].unsqueeze(3).unsqueeze(3)
        return x_pred, resampled_theta_0

    def norm_fa_from_sol(self, x_span, z_a):
        theta = x_span
        nb_observed = x_span.shape[1] if self.use_complete_signal else self.nb_observed
        if self.za_dim > 0 and not self.no_fa:
            norm_fa = self.fa(torch.cat((torch.sin(theta[:, :, :2, 0, 0]),
                                           torch.cos(theta[:, :, :2, 0, 0]),
                                           theta[:, :, 2:, 0, 0],
                                         z_a.unsqueeze(1).expand(-1, nb_observed, -1)), 2)).norm(2, dim=2).mean().item()
        else:
            norm_fa = 0.

        return norm_fa

    def constraint_traj_from_sol(self, t_eval, x, x_hat):
        x_hat = x_hat[:, :, :2]
        nb_observed = x.shape[1] if self.use_complete_signal else self.nb_observed
        diff_sin = (torch.sin(x[:, :nb_observed]) - torch.sin(x_hat[:, :nb_observed])) ** 2
        diff_cos = (torch.cos(x[:, :nb_observed]) - torch.cos(x_hat[:, :nb_observed])) ** 2
        l_traj = (diff_sin + diff_cos).mean(2).mean(1)
        return l_traj, t_eval

    def forward(self, t_span, x):
        z_e, z_a, _, _, _ = self.enc(t_span, x[:, :self.nb_observed])
        ode_f = lambda t, theta: self.ode_f(t, theta, z_a)
        x_pred_hybrid = odeint(ode_f, z_e.repeat(1, 2), t_span, **self.param_ode_solver).permute(1, 0, 2)[:, :, :2].unsqueeze(3).unsqueeze(3)
        return t_span, x_pred_hybrid

    def detailed_forward(self, t_span, x):
        z_e, z_a, q_z_e, q_z_a, x_clean = self.enc(t_span, x[:, :self.nb_observed])
        ode_f = lambda t, theta: self.ode_f(t, theta, z_a)
        mu_x_pred_tot = odeint(ode_f, z_e.repeat(1, 2), t_span, **self.param_ode_solver)
        x_pred_hybrid, x_pred_fp = torch.chunk(mu_x_pred_tot, 2, 2)
        x_pred_hybrid = x_pred_hybrid.permute(1, 0, 2)[:, :, :].unsqueeze(3).unsqueeze(3)
        x_pred_fp = x_pred_fp.permute(1, 0, 2)[:, :, :].unsqueeze(3).unsqueeze(3)
        return t_span, x_pred_hybrid, x_pred_fp, z_e, z_a, q_z_e, q_z_a, x_clean

    def predicted_parameters(self, t_span, x, zero_param=False):
        z_e, z_a, q_z_e, q_z_a, x_clean = self.enc(t_span, x[:, :self.nb_observed])
        mu_z_e, log_sigma_z_e = torch.chunk(q_z_e, 2, 1)

        return mu_z_e

    def predicted_parameters_as_dict(self, t_span, x, zero_param=False):
        mu_p = self.predicted_parameters(t_span, x, zero_param)
        return {"\\theta_0": mu_p[:, [0]], "\\theta_1": mu_p[:, [1]],
                "\\dot \\theta_0": mu_p[:, [2]], "\\dot \\theta_1": mu_p[:, [3]]}

    def loss(self, t_span, x):
        return self.loss_augm(t_span, x, None)

    def loss_augm(self, t_span, x, zp):
        b_size = x.shape[0]
        nb_steps = t_span.shape[0]

        nb_observed = x.shape[1] if self.use_complete_signal else self.nb_observed
        t_span, x_pred_hybrid, x_pred_fp, z_e, z_a, q_z_e, q_z_a, x_clean = self.detailed_forward(t_span, x[:, :nb_observed])
        mu_a, log_sigma_a = torch.chunk(q_z_a, 2, 1)
        sigma_a = torch.exp(log_sigma_a)
        x_pred_hybrid_all = x_pred_hybrid
        x_pred_hybrid = x_pred_hybrid[:, :, :2]
        x_pred_fp = x_pred_fp[:, :, :2]

        sigma_x_pred_tot_sin = torch.exp(self.sigma_x_sin).unsqueeze(0).unsqueeze(1).expand(b_size, nb_steps,
                                                                                            -1).unsqueeze(
            3).unsqueeze(3)
        sigma_x_pred_tot_cos = torch.exp(self.sigma_x_cos).unsqueeze(0).unsqueeze(1).expand(b_size, nb_steps,
                                                                                            -1).unsqueeze(
            3).unsqueeze(3)

        norm_fa = self.norm_fa_from_sol(x_pred_hybrid_all, mu_a)

        diff_sin = torch.distributions.Normal(loc=torch.sin(x_pred_hybrid[:, :nb_observed]),
                                              scale=sigma_x_pred_tot_sin).log_prob(torch.sin(x[:, :nb_observed])).sum(1).sum(1)
        diff_cos = torch.distributions.Normal(loc=torch.cos(x_pred_hybrid[:, :nb_observed]),
                                              scale=sigma_x_pred_tot_cos).log_prob(torch.cos(x[:, :nb_observed])).sum(1).sum(1)
        ll_traj = diff_sin + diff_cos

        KL_prior_posterior = kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a))

        # No term for z_p as we assume a uniform prior.
        ELBO = ll_traj - KL_prior_posterior

        # Regularizer for making the correction small.
        if self.alpha > 0.:
            bound_kl_physics_reg = (kl_gaussians(torch.flatten(torch.sin(x_pred_hybrid[:, :nb_observed]), 1),
                                                torch.flatten(sigma_x_pred_tot_sin, 1),
                                                torch.flatten(torch.sin(x_pred_fp[:, :nb_observed]), 1),
                                                              torch.flatten(sigma_x_pred_tot_sin, 1)) \
                                   + kl_gaussians(torch.flatten(torch.cos(x_pred_hybrid[:, :nb_observed]), 1),
                                                  torch.flatten(sigma_x_pred_tot_cos, 1),
                                                  torch.flatten(torch.cos(x_pred_fp[:, :nb_observed]), 1),
                                                  torch.flatten(sigma_x_pred_tot_cos, 1)) \
                                   + kl_gaussians(mu_a, sigma_a, torch.zeros_like(mu_a), torch.ones_like(sigma_a))).mean()
        else:
            bound_kl_physics_reg = 0.


        # Trying to make the two-step encoder as much related to the physics as possible
        if self.beta > 0.:
            x_pred_fp_detached = x_pred_fp.detach()

            x_pred_fp_detached_masked = torch.cat((x_pred_fp_detached[:, -self.nb_observed_theta_0:, 0],
                                                   x_pred_fp_detached[:, -self.nb_observed_theta_1:, 1]), 1).reshape(b_size, -1)

            x_pred_fp_detached_masked_sin_cos_encoded = torch.cat((torch.sin(x_pred_fp_detached_masked), torch.cos(x_pred_fp_detached_masked)), 1)
            R_da_1 = ((x_clean - x_pred_fp_detached_masked_sin_cos_encoded) ** 2).sum(1).mean()
        else:
            R_da_1 = 0.

        # Forcing the second step to find the correct latent variables
        if self.gamma > 0.:
            resampled_zp = torch.rand(b_size, self.zp_dim, device=self.device) * (self.max_zp - self.min_zp) \
                           + self.min_zp

            mu_x_pred_fp = odeint(self.fp, resampled_zp, t_span[:self.nb_observed], **self.param_ode_solver)

            x_r_detached = mu_x_pred_fp.detach().requires_grad_(True).permute(1, 0, 2)[:, :, :2].unsqueeze(3).unsqueeze(3)
            x_r_detached_masked = torch.cat((x_r_detached[:, -self.nb_observed_theta_0:, 0],
                                             x_r_detached[:, -self.nb_observed_theta_1:, 1]), 1).reshape(b_size, -1)

            x_r_detached_masked_sin_cos_encoded = torch.cat((torch.sin(x_r_detached_masked),
                                                             torch.cos(x_r_detached_masked)), 1)
            mu_p, log_sigma_p = torch.chunk(self.enc.enc_ze(x_r_detached_masked_sin_cos_encoded), 2, 1)

            R_da_2 = ((mu_p - resampled_zp) ** 2).sum(1).mean()
        else:
            R_da_2 = 0.

        loss = -ELBO.mean() + self.alpha * bound_kl_physics_reg + self.beta * R_da_1 + self.gamma * R_da_2
        return loss, torch.tensor(norm_fa), torch.tensor(-1.)

