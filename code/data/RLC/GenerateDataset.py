#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

from code.simulators import RLCCircuit
import torch
from tqdm import tqdm
import os
import pickle


def gen_data(n=500, shifted="False"):
    if shifted == "small_all":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(3., 4.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(3., 5.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "small_R":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(3., 4.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "medium_all":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(4., 8.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(3., 5.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "medium_R":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(4., 8.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "large_all":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(10., 20.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(3., 5.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "large_R":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(10., 20.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    elif shifted == "None":
        distributions = {
            "R": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(3., 5.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(5., 1.5).sample_n(x)
        }
    else:
        distributions = {
            "R": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "L": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "C": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x),
            "V_a": lambda x: torch.distributions.Uniform(1.5, 3.5).sample_n(x),
            "V_c": lambda x: torch.distributions.Uniform(.5, 2.5).sample_n(x),
            "omega": lambda x: torch.distributions.Uniform(1., 2.5).sample_n(x)
        }
    s = RLCCircuit()
    s.n_timesteps = 200
    s.T0 = 0.
    s.T1 = 20.
    dataset = torch.zeros(n, s.n_timesteps + 1, 1, 2)
    true_param = {"R": torch.zeros(n),
                  "L": torch.zeros(n),
                  "C": torch.zeros(n),
                  "V_a": torch.zeros(n),
                  "V_c": torch.zeros(n),
                  "omega": torch.zeros(n)}
    for i in tqdm(range(n)):
        true_param['R'][i] = distributions["R"](1)[0]
        true_param['L'][i] = distributions["L"](1)[0]
        true_param['C'][i] = distributions["C"](1)[0]
        true_param['V_a'][i] = 2.5
        true_param['V_c'][i] = 1.
        true_param['omega'][i] = 2.
        t, x = s.sample_sequences({x: t[i] for x, t in true_param.items()})
        dataset[i, :, :, :] = x
    dataset = dataset.permute(0, 1, 3, 2).unsqueeze(3)
    return t, dataset, true_param


path = 'code/data/RLC'
if not os.path.exists(path):
    os.makedirs(path)

with open(r"%s/train.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(3000), output_file)

with open(r"%s/valid.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(1000), output_file)

with open(r"%s/test_shifted.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(1000, "None"), output_file)

for hardness in ["small", "medium", "large"]:
    for shift in ["all", "R"]:
        shift_def = hardness + "_" + shift
        with open(r"%s/%s_test.pkl" % (path, shift_def), "wb") as output_file:
            pickle.dump(gen_data(1000, shifted=shift_def), output_file)