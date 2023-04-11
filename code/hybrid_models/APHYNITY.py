#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import math
import pickle

import torch.nn as nn
from code.simulators import PhysicalModel, NoSimulator
import torch
from code.nn import MLP, ConditionalUNet, Permute, act_dict
from torchdiffeq import odeint
from code.hybrid_models.HybridAutoencoder import HybridAutoencoder


class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=x_dim, hidden_size=hidden_size, batch_first=True)
        self.mu, self.sigma = None, None

    def forward(self, t, x):
        _, out = self.rnn(x.squeeze(4).squeeze(3))
        return out[0]


class DoublePendulumEncoder(nn.Module):
    def __init__(self, layers=[300, 300, 300], za_dim=50, ze_dim=4, initial_guess=True,
                 nb_observed_theta_0=25, nb_observed_theta_1=25, obtain_init_position=False):
        super(DoublePendulumEncoder, self).__init__()
        self.za_dim = za_dim
        self.ze_dim = ze_dim
        in_dim = 2
        self.obtain_init_position = obtain_init_position
        self.initial_guess = initial_guess
        self.total_time = nb_observed_theta_0 + nb_observed_theta_1
        self.nb_observed_theta_0 = nb_observed_theta_0
        self.nb_observed_theta_1 = nb_observed_theta_1

        lis = [self.total_time*in_dim] + layers
        los = layers + [za_dim + ze_dim]
        layers = []
        for li, lo in zip(lis, los):
            layers += [nn.Linear(li, lo), nn.SELU()]
        layers.pop()
        self.encoder = nn.Sequential(*layers)

    def forward(self, t, x):
        in_s = t.shape[0]
        frequency = 1 / t[1]
        if self.total_time < 50 or True:
            x_masked = torch.cat((x[:, -self.nb_observed_theta_0:, 0], x[:, -self.nb_observed_theta_1:, 1]), 1)
        else:
            dx, dy = x[:, 0, :], x[:, 2, :]
            diff = (dx - dy).unsqueeze(2)
            choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
            _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
            omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) * frequency / 2
            init_state = torch.cat((x[:, 1, :], -omegas), 1)
            x_masked = x.view(x.shape[0], -1)

        sin_cos_encoded = torch.cat((torch.sin(x_masked), torch.cos(x_masked)), 1)
        z_all = self.encoder(sin_cos_encoded.reshape(x_masked.shape[0], -1))

        if self.initial_guess and self.total_time == 50:
            z_e = init_state + z_all[:, :self.ze_dim] #* 0.1
        elif self.obtain_init_position:
            z_e = torch.cat([x[:, 0, :2, 0, 0], z_all[:, 2:self.ze_dim]], 1)
        else:
            z_e = z_all[:, :self.ze_dim]
        z_a = None if self.za_dim == 0 else z_all[:, self.ze_dim:]
        return z_e, z_a


class HybridDecoder(nn.Module):
    def __init__(self, fp: PhysicalModel, fp_param_converter_hidden=None, fp_param_converter_act="SELU",
                 fp_param_converter_final_act="Softplus", fa_hidden=None, fa_hidden_act="SELU", fa_final_act=None,
                 encoder_dim=128, **kwargs):
        super(HybridDecoder, self).__init__()

        # Create net that maps hidden from encoder to physical parameters.
        if fp_param_converter_hidden is None:
            fp_param_converter_hidden = 4 * [300]
        layers_fp = [encoder_dim] + fp_param_converter_hidden + [len(fp.incomplete_param_dim_textual)]
        self.fp_param_converter = MLP(layers_fp, hidden_act=fp_param_converter_act,
                                      final_act=fp_param_converter_final_act)

        # Create net that maps hidden from encoder + state x to fa(x;h) contribution.
        if fa_hidden is None:
            fa_hidden = [300] * 4
        x_dim = fp._X_dim
        layers_fa = [encoder_dim + x_dim] + fa_hidden + [x_dim]
        self.fa = MLP(layers_fa, hidden_act=fa_hidden_act, final_act=fa_final_act)

        self.fp = fp

        self.h = None
        self.fp_params = None

    def to(self, device):
        super(HybridDecoder, self).to(device)
        self.fp.to(device)
        self.fa.to(device)
        self.fp_param_converter.to(device)
        return self

    def set_hidden(self, h):
        # h: B_size x hidden_size
        self.h = h

    def forward_fa(self, t, x):
        self.fa(torch.cat((x, self.h), 1))

        return self.fa(torch.cat((x, self.h), 1))

    def get_physical_parameters(self, x, zero_param=False, zp=None, as_dict=True):
        fp_params = self.fp_param_converter(self.h) if zp is None else zp
        physical_params = {}
        if not as_dict:
            return fp_params
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)
        if zero_param:
            for i, p in enumerate(self.fp.missing_param_dim_textual):
                physical_params[p] = torch.zeros(x.shape[0], device=x.device).unsqueeze(1)

        return physical_params

    def forward_fp(self, t, x, zp=None):
        physical_params = self.get_physical_parameters(x) if zp is None else zp
        return self.fp.parameterized_forward(t, x, **physical_params)

    def forward(self, t, x, zp=None):
        if x.shape[0] != self.h.shape[0]:
            raise Exception(
                "Mismatch between hidden state batch size %d and x batch size %d." % (self.h.shape[0], x.shape[0]))

        dx = self.forward_fa(t, x) + self.forward_fp(t, x, zp)
        return dx#.unsqueeze(1)


class APHYNITYAutoencoder(HybridAutoencoder):
    def __init__(self, fp, augmented=False, zp_priors=None, device="cpu", **config):
        super(APHYNITYAutoencoder, self).__init__()
        encoder_out = 128
        self.enc = Encoder(fp._X_dim, encoder_out).to(device)
        param_decoder = {"fp": fp.to(device),
                         "fp_param_converter_hidden_w": 200,
                         "fp_param_converter_hidden_n": 3,
                         "fp_param_converter_act": "ReLU",
                         "fp_param_converter_final_act": "Softplus",
                         "fa_hidden_w": 200,
                         "fa_hidden_n": 3,
                         "fa_hidden_act": "ReLU",
                         "fa_final_act": None,
                         "encoder_dim": encoder_out}

        self.lambda_p = config.get("lambda_p", float("nan"))

        for k, v in param_decoder.items():
            if k[:2] == "fa":
                if k[3:] in config.get("fa", {}):
                    param_decoder[k] = config["fa"][k[3:]]
            elif k in config:
                param_decoder[k] = config[k]

        param_decoder["fa_hidden"] = [param_decoder["fa_hidden_w"]] * param_decoder["fa_hidden_n"]
        param_decoder["fp_param_converter_hidden"] = [param_decoder["fp_param_converter_hidden_w"]] * param_decoder["fp_param_converter_hidden_n"]

        self.dec = HybridDecoder(**param_decoder).to(device)
        self.nb_observed = 50
        self.device = device
        self.augmented = augmented
        self.min_zp = None
        if zp_priors is not None and self.augmented:
            min_zp = []
            max_zp = []
            for k, v in zp_priors.items():
                min_zp.append(v["min"])
                max_zp.append(v["max"])
            self.min_zp = torch.tensor(min_zp, device=device)
            self.max_zp = torch.tensor(max_zp, device=device)

    def to(self, device):
        self.device = device
        self.enc.to(device)
        self.dec.to(device)
        if self.min_zp is not None:
            self.min_zp.to(device)
            self.max_zp.to(device)
        return self

    def forward(self, t_span, x):
        h = self.enc(t_span[:self.nb_observed], x[:, :self.nb_observed])
        self.dec.set_hidden(h)
        x_pred = odeint(self.dec, x[:, 0, :, 0, 0], t_span, atol=1e-5, rtol=1e-5)
        return t_span, x_pred.permute(1, 0, 2).unsqueeze(3).unsqueeze(4)

    def augmented_data(self, t_span, x):
        h = self.enc(t_span[:self.nb_observed], x[:, :self.nb_observed])
        self.dec.set_hidden(h)

        resampled_zp = torch.rand(x.shape[0], self.min_zp.shape[0], device=self.device) * (self.max_zp - self.min_zp) \
                       + self.min_zp
        fp = self.dec.get_physical_parameters(None, None, resampled_zp)
        with torch.no_grad():
            dec = lambda t, x: self.dec(t, x, fp)
            x_pred = odeint(dec, x[:, 0, :, 0, 0], t_span, atol=1e-5, rtol=1e-5)
        return x_pred.permute(1, 0, 2).unsqueeze(3).unsqueeze(4), resampled_zp

    def predicted_parameters(self, t_span, x, zero_param=False):
        h = self.enc(t_span[:self.nb_observed], x[:, :self.nb_observed])
        self.dec.set_hidden(h)
        return self.dec.get_physical_parameters(x, zero_param, as_dict=False)

    def predicted_parameters_as_dict(self, t_span, x, zero_param=False) -> dict:
        h = self.enc(t_span[:self.nb_observed], x[:, :self.nb_observed])
        self.dec.set_hidden(h)
        return self.dec.get_physical_parameters(x, zero_param, as_dict=True)

    def penalty(self, t_eval, x_span):
        x = x_span[:, :self.nb_observed, :, 0, 0]
        concat_x_h = torch.cat((self.dec.h.unsqueeze(1).expand(-1, x.shape[1], -1), x), 2)
        l = (self.dec.fa(concat_x_h).norm(2, dim=2)) ** 2
        return l.mean(1)

    def constraint_traj(self, t_span, x):
        t_eval, x_hat = self.forward(t_span, x[:, :self.nb_observed])
        l_traj = (x[:self.nb_observed] - x_hat[:self.nb_observed]).norm(2, dim=2).mean(0)
        return l_traj, t_eval

    def lagrangian(self, lambda_p, t_span, x):
        l_trajectory, t_eval = self.constraint_traj(t_span, x)
        l_penalty = self.penalty(t_eval, x).mean() ** .5
        return l_penalty + lambda_p * l_trajectory.mean(), l_penalty.detach(), l_trajectory.mean().detach()

    def loss(self, t_span, x) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self.lagrangian(self.lambda_p, t_span, x)


class APHYNITYAutoencoderDoublePendulum(HybridAutoencoder):
    def __init__(self, fp, augmented=False, zp_priors=None, device="cpu", **config):
        super(APHYNITYAutoencoderDoublePendulum, self).__init__()
        self.ze_dim = 4
        self.lambda_p = config.get("lambda_p", float("nan"))
        self.nb_observed = config.get("nb_observed", 25)
        self.nb_observed_theta_0 = config.get("nb_observed_theta_0", 25)
        self.nb_observed_theta_1 = config.get("nb_observed_theta_1", 25)
        self.za_dim = config.get("za_dim", 5)
        self.cos_sin_encoding = config.get("cos_sin_encoding", False)
        self.weight_penalty = config.get("weight_penalty", False)
        self.weight_constraint = config.get("weight_constraint", False)
        self.use_complete_signal = config.get("use_complete_signal", False)
        self.partial_observability = config.get("partial_observability", False)
        self.initial_guess = config.get("initial_guess", True)
        self.simplified_fa = config.get("simplified_fa", False)
        self.obtain_init_position = config.get("obtain_init_position", False)
        layers_encoder = [config.get("hidden_size_encoder", 300)] * config.get("nb_layers_encoder", 3)

        self.enc = DoublePendulumEncoder(layers=layers_encoder, za_dim=self.za_dim, ze_dim=self.ze_dim,
                                         nb_observed_theta_0=self.nb_observed_theta_0,
                                         nb_observed_theta_1=self.nb_observed_theta_1,
                                         obtain_init_position=self.obtain_init_position).to(device)
        self.fp = fp.to(device)
        self.no_fa = config.get("no_fa", False)
        self.no_fp = config.get("no_fp", False)
        self.no_APHYNITY = config.get("no_APHYNITY", False)
        if self.no_fp:
            self.fp = NoSimulator()

        if self.no_fa:
            self.dec = lambda x: torch.zeros(list(x.shape[:-1]) + [self.ze_dim], device=x.device)
        elif self.simplified_fa:
            self.linear_layer = nn.Linear(8, self.ze_dim)
            self.cos_sin_encoding = False
            self.dec = lambda x: self.linear_layer(torch.cat((torch.sin(x[..., :2]), torch.cos(x[..., :2]),
                                                              x[..., 2:]/10, (x[..., 2:]**2)/100), -1))
        else:
            in_dim = self.ze_dim if not self.cos_sin_encoding else self.ze_dim + 2

            layers_fa = [config.get("hidden_size_fa", 300)] * config.get("nb_layers_fa", 3)
            lis = [in_dim + self.za_dim] + layers_fa
            los = layers_fa + [self.ze_dim]
            layers = []
            for li, lo in zip(lis, los):
                layers += [nn.Linear(li, lo), nn.SELU()]
            layers.pop()
            self.dec = nn.Sequential(*layers)

        self.device = device
        self.augmented = augmented
        self.min_zp = None
        if zp_priors.get("path", None):
            with open(zp_priors.get("path", None), "rb") as output_file:
                self.init_states = pickle.load(output_file)
        else:
            self.init_states = None
            if zp_priors is not None and self.augmented:
                min_zp = []
                max_zp = []
                for k, v in zp_priors.items():
                    min_zp.append(v["min"])
                    max_zp.append(v["max"])
                self.min_zp = torch.tensor(min_zp, device=device)
                self.max_zp = torch.tensor(max_zp, device=device)

    def to(self, device):
        self.device = device
        self.enc.to(device)
        if not self.no_fa and not self.simplified_fa:
            self.dec.to(device)
        self.fp.to(device)
        if self.min_zp is not None:
            self.min_zp.to(device)
            self.max_zp.to(device)
        return self

    def ode_f(self, z_e, z_a):
        if self.za_dim > 0:
            if self.cos_sin_encoding:
                return lambda t, theta: self.fp(t, theta) + self.dec(torch.cat((torch.sin(theta[:, :2]),
                                                                                torch.cos(theta[:, :2]),
                                                                                theta[:, 2:], z_a), 1))

            return lambda t, theta: self.fp(t, theta) + self.dec(torch.cat((theta, z_a), 1))

        if self.simplified_fa:
            return lambda t, theta: self.fp(t, theta) + self.dec(theta)

        if self.cos_sin_encoding:
            return lambda t, theta: self.fp(t, theta) + self.dec(torch.cat((torch.sin(theta[:, :2]),
                                                                            torch.cos(theta[:, :2]),
                                                                            theta[:, 2:]), 1))

        return lambda t, theta: self.fp(t, theta) + self.dec(theta)

    def forward(self, t_span, x):
        z_e, z_a = self.enc(t_span, x[:, :self.nb_observed])
        x_pred = odeint(self.ode_f(z_e, z_a), z_e, t_span, atol=1e-5,
                        rtol=1e-5).permute(1, 0, 2)[:, :, :2].unsqueeze(3).unsqueeze(3)

        return t_span, x_pred

    def augmented_data(self, t_span, x):
        theta_0, z_a = self.enc(t_span, x[:, :self.nb_observed])

        if self.init_states is not None:
            idx = torch.randperm(self.init_states.shape[0])[:x.shape[0]]
            resampled_theta_0 = self.init_states[idx] + torch.randn_like(self.init_states[idx]) * .1
        else:
            resampled_theta_0 = torch.rand(x.shape[0], self.min_zp.shape[0], device=self.device) * \
                                (self.max_zp - self.min_zp) + self.min_zp
        with torch.no_grad():
            dec = self.ode_f(resampled_theta_0, z_a)
            x_pred = odeint(dec, resampled_theta_0, t_span, atol=1e-5,
                            rtol=1e-5).permute(1, 0, 2)[:, :, :2].unsqueeze(3).unsqueeze(3)
        return x_pred, resampled_theta_0

    def predicted_parameters_as_dict(self, t_span, x, zero_param=False):
        z_e, z_a = self.enc(t_span, x[:, :self.nb_observed])
        return {"\\theta_0": z_e[:, [0]], "\\theta_1": z_e[:, [1]],
                "\\dot \\theta_0": z_e[:, [2]], "\\dot \\theta_1": z_e[:, [3]]}

    def predicted_parameters(self, t_span, x, zero_param=False):
        z_e, z_a = self.enc(t_span, x[:, :self.nb_observed])
        return z_e

    def penalty(self, t_eval, x_span):
        raise NotImplementedError

    def penalty_from_sol(self, x_span, z_a):
        nb_observed = x_span.shape[1] if self.use_complete_signal else self.nb_observed

        x = x_span[:, :nb_observed]
        if self.za_dim > 0:
            z_a = z_a.unsqueeze(1).expand(-1, nb_observed, -1)
            if self.cos_sin_encoding:
                l = ((self.dec(torch.cat((torch.sin(x[:, :, :2, 0, 0]),
                                          torch.cos(x[:, :, :2, 0, 0]),
                                          x[:, :, 2:, 0, 0], z_a), 2))).norm(2, dim=2)) #** 2
            else:
                l = ((self.dec(torch.cat((x, z_a), 2))).norm(2, dim=2)) #** 2
        elif self.cos_sin_encoding:
            l = ((self.dec(torch.cat((torch.sin(x[:, :, :2, 0, 0]),
                                      torch.cos(x[:, :, :2, 0, 0]),
                                      x[:, :, 2:, 0, 0]), 2))).norm(2, dim=2)) #** 2
        else:
            l = ((self.dec(x)).norm(2, dim=2)) #** 2

        if self.weight_penalty:
            weights = 1.25 ** -torch.arange(l.shape[1])
            weights = weights.sum()
            l = l * weights.unsqueeze(0)
            return l.sum(1)

        return l.mean(1)

    def constraint_traj(self, t_span, x):
        nb_observed = x.shape[1] if self.use_complete_signal else self.nb_observed
        t_eval, x_hat = self.forward(t_span, x[:, :nb_observed])
        x_hat = x_hat[:, :, :2]
        diff_sin = (torch.sin(x[:, :nb_observed]) - torch.sin(x_hat[:, :nb_observed]))**2
        diff_cos = (torch.cos(x[:, :nb_observed]) - torch.cos(x_hat[:, :nb_observed]))**2
        if self.weight_constraint:
            l = (diff_sin + diff_cos).mean(2)
            weights = 1.25 ** -torch.arange(l.shape[1])
            weights = weights / weights.sum()
            l = l * weights.unsqueeze(0)
            return l.sum(1)
        l_traj = (diff_sin + diff_cos).mean(2).mean(1)
        return l_traj, t_eval

    def constraint_traj_from_sol(self, t_eval, x, x_hat):
        x_hat = x_hat[:, :, :2]
        nb_observed = x.shape[1] if self.use_complete_signal else self.nb_observed
        diff_sin = (torch.sin(x[:, :nb_observed]) - torch.sin(x_hat[:, :nb_observed]))**2
        diff_cos = (torch.cos(x[:, :nb_observed]) - torch.cos(x_hat[:, :nb_observed]))**2
        l_traj = (diff_sin + diff_cos).mean(2).mean(1)
        return l_traj, t_eval

    def lagrangian(self, lambda_p, t_span, x):
        z_e, z_a = self.enc(t_span, x[:, :self.nb_observed])
        x_pred = odeint(self.ode_f(z_e, z_a), z_e, t_span, atol=1e-5, rtol=1e-5).permute(1, 0, 2).unsqueeze(3).unsqueeze(3)
        l_trajectory, t_eval = self.constraint_traj_from_sol(t_span, x, x_pred)
        l_trajectory = l_trajectory ** .5
        l_penalty = self.penalty_from_sol(x_pred, z_a).mean()

        if self.augmented and False:
            x_pred, _ = self.augmented_data(t_span, x)
            self.augmented = False
            augm_loss, _ = self.lagrangian(lambda_p, t_span, x_pred)
            self.augmented = True
            return augm_loss, l_penalty.detach()

        if self.no_APHYNITY:
            return l_trajectory.mean(), l_penalty.detach(), l_trajectory.mean().detach()
        return l_penalty + lambda_p * l_trajectory.mean(), l_penalty.detach(), l_trajectory.mean().detach()

    def loss(self, t_span, x):
        return self.lagrangian(self.lambda_p, t_span, x)

    def lagrangian_augm(self, lambda_p, t_span, x, zp=None):
        return self.lagrangian(lambda_p, t_span, x)


class APHYNITYAutoencoderReactionDiffusion(HybridAutoencoder):
    def __init__(self, fp, augmented=False, zp_priors=None, device="cpu", **config):
        super(APHYNITYAutoencoderReactionDiffusion, self).__init__()
        self.dim_in = 2
        self.nb_observed = 10
        self.dim_za, self.dim_zp = 10, 2
        self.lambda_p = config.get("lambda_p", float("nan"))
        self.enc = nn.Sequential(nn.Flatten(0, 1), nn.Conv2d(self.dim_in, 16, 3), nn.ReLU(),
                            nn.Conv2d(16, 32, 3), nn.AvgPool2d(2),
                            nn.Conv2d(32, 64, 3), nn.ReLU(),
                            nn.Conv2d(64, 64, 3), nn.AvgPool2d(2),
                            nn.Conv2d(64, 32, 3), nn.ReLU(), nn.Unflatten(0, (-1, self.nb_observed)),
                            Permute((0, 2, 1, 3, 4)), nn.Conv3d(32, 16, 2), nn.ReLU(),
                            nn.Conv3d(16, 16, 2), nn.Flatten(1, 4),
                            nn.Linear(128, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, self.dim_za + self.dim_zp))

        self.fp_param_converter_final_act = act_dict["ReactionDiffusionParametersScaler"]
        self.fa = nn.Sequential(nn.Conv2d(2 + self.dim_za, 16, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(16, 2, 3, padding=1))
        self.fp = fp

        self.device = device
        self.augmented = augmented
        if augmented:
            min_zp = []
            max_zp = []
            for k, v in zp_priors.items():
                min_zp.append(v["min"])
                max_zp.append(v["max"])
            self.min_zp = torch.tensor(min_zp, device=device)
            self.max_zp = torch.tensor(max_zp, device=device)

    def forward_step(self, t, x, zp, za):
        x_fa = torch.cat((x, za.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])), 1)
        return self.fa(x_fa) + self.fp.parameterized_forward(t, x, **zp)

    def to(self, device):
        self.device = device
        self.enc.to(device)
        self.fa.to(device)
        self.fp.to(device)
        if self.augmented:
            self.min_zp.to(device)
            self.max_zp.to(device)
        return self

    def get_physical_parameters(self, h, zero_param=False, as_dict=False):
        za, zp = h[:, :self.dim_za], h[:, self.dim_za:]
        zp = self.fp_param_converter_final_act(zp)

        fp_params = zp
        if not as_dict:
            return fp_params, za
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)
        if zero_param:
            for i, p in enumerate(self.fp.missing_param_dim_textual): \
                physical_params[p] = torch.zeros(h.shape[0], device=h.device).unsqueeze(1)

        return physical_params, za

    def forward(self, t_span, x, h=None):
        if h is None:
            h = self.enc(x[:, :self.nb_observed])
        physical_params, za = self.get_physical_parameters(h, as_dict=True)
        x_pred = odeint(lambda t, x: self.forward_step(t, x, physical_params, za),
                        x[:, 0], t_span, atol=1e-5, rtol=1e-5).permute(1, 0, 2, 3, 4)
        return t_span, x_pred

    def augmented_data(self, t_span, x):
        h = self.enc(x[:, :self.nb_observed])
        physical_params, za = self.get_physical_parameters(h)
        resampled_zp = torch.rand(x.shape[0], self.min_zp.shape[0], device=self.device) * (self.max_zp - self.min_zp) \
                       + self.min_zp
        fp_params = resampled_zp
        physical_params = {}
        for i, p in enumerate(self.fp.incomplete_param_dim_textual):
            physical_params[p] = fp_params[:, i].unsqueeze(1)
        with torch.no_grad():
            x_pred = odeint(lambda t, x: self.forward_step(t, x, physical_params, za),
                            x[:, 0], t_span, atol=1e-5, rtol=1e-5).permute(1, 0, 2, 3, 4)
        return x_pred, resampled_zp

    def predicted_parameters(self, t_span, x, zero_param=False):
        h = self.enc(x[:, :self.nb_observed])
        physical_params, za = self.get_physical_parameters(h, zero_param=False)
        return physical_params

    def predicted_parameters_as_dict(self, t_span, x, zero_param=False) -> dict:
        h = self.enc(x[:, :self.nb_observed])
        physical_params, za = self.get_physical_parameters(h, zero_param=False, as_dict=True)
        return physical_params

    def penalty(self, t_eval, x_span, h=None):
        if h is None:
            h = self.enc(x_span[:, :self.nb_observed])
        x = x_span[:, :self.nb_observed]
        physical_params, za = self.get_physical_parameters(h)
        za = za.unsqueeze(1).expand(-1, self.nb_observed, -1).reshape(-1, self.dim_za)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x_fa = torch.cat((x, za.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])), 1)
        response = self.fa(x_fa).reshape(x_span.shape[0], self.nb_observed, -1)
        l = (response.norm(2, dim=2)) ** 2
        return l.mean(1)

    def constraint_traj(self, t_span, x, h=None):
        t_eval, x_hat = self.forward(t_span, x[:, :self.nb_observed], h)
        l_traj = (x[:, :self.nb_observed] - x_hat[:, :self.nb_observed]).view(x.shape[0],
                                                                              self.nb_observed,
                                                                              -1).norm(2, dim=2).mean(1)
        return l_traj, t_eval

    def lagrangian(self, lambda_p, t_span, x):
        h = self.enc(x[:, :self.nb_observed])
        l_trajectory, t_eval = self.constraint_traj(t_span, x, h)
        l_penalty = self.penalty(t_eval, x, h).mean() ** .5
        return l_penalty + lambda_p * l_trajectory.mean(), l_penalty.detach(), l_trajectory.mean().detach()

    def loss(self, t_span, x) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return self.lagrangian(self.lambda_p, t_span, x)
