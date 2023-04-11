#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

from code.simulators import ReactionDiffusion
import torch
from tqdm import tqdm
import os
import pickle


def gen_data(n=500, shifted=""):
    if shifted == "small_all":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(2e-3, 4e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(1e-3, 1e-2).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(5e-3, 8e-3).sample_n(x)
        }
    elif shifted == "small_k":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(1e-3, 2e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(3e-3, 7e-3).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(5e-3, 8e-3).sample_n(x)
        }
    elif shifted == "medium_all":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(2e-3, 4e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(1e-3, 1e-2).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(8e-3, 2e-2).sample_n(x)
        }
    elif shifted == "medium_k":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(1e-3, 2e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(3e-3, 7e-3).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(8e-3, 2e-2).sample_n(x)
        }
    elif shifted == "large_all":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(2e-3, 4e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(1e-3, 1e-2).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(2e-2, 1e-1).sample_n(x)
        }
    elif shifted == "large_k":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(1e-3, 2e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(3e-3, 7e-3).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(2e-2, 1e-1).sample_n(x)
        }
    elif shifted == "None":
        distributions = {
            "a": lambda x: torch.distributions.Uniform(2e-3, 4e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(1e-3, 1e-2).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(3e-3, 5e-3).sample_n(x)
        }
    else:
        distributions = {
            "a": lambda x: torch.distributions.Uniform(1e-3, 2e-3).sample_n(x),
            "b": lambda x: torch.distributions.Uniform(3e-3, 7e-3).sample_n(x),
            "k": lambda x: torch.distributions.Uniform(3e-3, 5e-3).sample_n(x)
        }
    s = ReactionDiffusion()
    s.n_timesteps = 50
    s.T0 = 0.
    s.T1 = 5.
    dataset = torch.zeros((n, s.n_timesteps + 1) + s._X_dim)
    true_param = {"a": torch.zeros(n),
                  "b": torch.zeros(n),
                  "k": torch.zeros(n)}
    for i in tqdm(range(n)):
        true_param['a'][i] = distributions["a"](1)[0]
        true_param['b'][i] = distributions["b"](1)[0]
        true_param['k'][i] = distributions["k"](1)[0]
        t, x = s.sample_sequences({x: t[i] for x, t in true_param.items()})
        dataset[i] = x.squeeze(1)

    return t, dataset, true_param


path = 'code/data/ReactionDiffusion'
if not os.path.exists(path):
    os.makedirs(path)

with open(r"%s/train.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(2000), output_file)

with open(r"%s/valid.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(100), output_file)

with open(r"%s/test_shifted.pkl" % path, "wb") as output_file:
    pickle.dump(gen_data(100, "None"), output_file)

for hardness in ["small", "medium", "large"]:
    for shift in ["all", "k"]:
        shift_def = hardness + "_" + shift
        with open(r"%s/%s_test.pkl" % (path, shift_def), "wb") as output_file:
            pickle.dump(gen_data(100, shifted=shift_def), output_file)
