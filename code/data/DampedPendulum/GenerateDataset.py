#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

from code.simulators import DampedPendulum
import torch
from tqdm import tqdm
import os
import pickle
import argparse


def gen_data(n=500, shifted=""):
    if shifted == "small_all":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(.3, .6).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x)
        }
    elif shifted == "small_alpha":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(.3, .6).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(1.5, 3.1).sample_n(x)
        }
    elif shifted == "medium_all":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(.6, 1.).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x)
        }
    elif shifted == "medium_alpha":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(.6, 1.).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(1.5, 3.1).sample_n(x)
        }
    elif shifted == "large_all":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x)
        }
    elif shifted == "large_alpha":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(1., 3.).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(1.5, 3.1).sample_n(x)
        }
    elif shifted == "None":
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(0., .6).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(.5, 1.5).sample_n(x)
        }
    else:
        distributions = {
            "alpha": lambda x: torch.distributions.Uniform(0., 0.6).sample_n(x),
            "omega_0": lambda x: torch.distributions.Uniform(1.5, 3.1).sample_n(x)
        }
    s = DampedPendulum()
    s.n_timesteps = 200
    s.T0 = 0.
    s.T1 = 20
    dataset = torch.zeros(n, s.n_timesteps + 1, 1, 2)
    true_param = {'omega_0': torch.zeros(n), 'alpha': torch.zeros(n), 'A': torch.zeros(n), 'phi': torch.zeros(n)}
    for i in tqdm(range(n)):
        true_param['omega_0'][i] = distributions["omega_0"](1)[0]
        true_param['alpha'][i] = distributions["alpha"](1)[0]
        true_param['phi'][i] = 0.
        true_param['A'][i] = 0.
        with torch.no_grad():
            t, x = s.sample_sequences({'omega_0': true_param['omega_0'][i],
                                       'alpha': true_param['alpha'][i],
                                       'A': true_param['A'][i],
                                       'phi': true_param['phi'][i]})
        dataset[i, :, :, :] = x
    dataset = dataset.permute(0, 1, 3, 2).unsqueeze(3)
    return t, dataset, true_param

if __name__ == "__main__":
    try:
        nb_data_train = 1000
    except:
        # Create the parser
        parser = argparse.ArgumentParser()
        # Add an argument
        parser.add_argument('--nb_train', type=int, default=1000)
        # Parse the argument
        args = parser.parse_args()

        nb_data_train = args.nb_train
    path = 'code/data/DampedPendulum'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(r"%s/train.pkl" % path, "wb") as output_file:
        pickle.dump(gen_data(nb_data_train), output_file)

    with open(r"%s/valid.pkl" % path, "wb") as output_file:
        pickle.dump(gen_data(100), output_file)

    with open(r"%s/test.pkl" % path, "wb") as output_file:
        pickle.dump(gen_data(100), output_file)

    with open(r"%s/valid_shifted.pkl" % path, "wb") as output_file:
        pickle.dump(gen_data(100, shifted="None"), output_file)

    with open(r"%s/test_shifted.pkl" % path, "wb") as output_file:
        pickle.dump(gen_data(100, shifted="None"), output_file)

    for hardness in ["small", "medium", "large"]:
        for shift in ["all", "alpha"]:
            shift_def = hardness + "_" + shift
            with open(r"%s/%s_test.pkl" % (path, shift_def), "wb") as output_file:
                pickle.dump(gen_data(100, shifted=shift_def), output_file)