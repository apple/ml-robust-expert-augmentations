#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import shutil

import torch
from matplotlib import pyplot as plt
from torch import nn

from code.hybrid_models.APHYNITY import APHYNITYAutoencoderDoublePendulum
from code.simulators import PhysicalModel
from code.simulators import DampedPendulum, RLCCircuit, ReactionDiffusion, DoublePendulum
from code.hybrid_models import APHYNITYAutoencoder, APHYNITYAutoencoderReactionDiffusion, HybridVAEDoublePendulum, \
    HybridVAEReactionDiffusion, HybridVAE
import yaml
from code.utils import plot_curves_partial, load_data, plot_curves_complete, plot_curves_double_pendulum
import torch.utils.data as data_utils
from tqdm import tqdm
import os
import math
import argparse
from datetime import datetime
from code.utils.utils import *



def run_exp(s: PhysicalModel, verbose=False, config=None, solver="None", config_name="", data_path=None,
            save_path=None):
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "cpu" #"mps" commented because conv3d not supported yet.
    else:
        device = "cpu"

    if config is None:
        raise Exception("No config provided.")

    param_text = s.incomplete_param_dim_textual
    # Loading data
    data_path = 'code/data/%s' % s.name if data_path is None else data_path
    print("Loading data in %s" % data_path)
    (t_train, x_train, true_param_train), dl_train, \
    (t_valid, x_valid, true_param_valid), \
    (t_test_shifted, x_test_shifted, true_param_test_shifted) = load_data(data_path, device)

    # Option to enforce training with reduced timeframe for the double pendulum
    # Training with long sequences is hard because of the chaotic behaviors of the double pendulum equations.
    if config.get("reduced_time_frame", False):
        print("training with fewer steps ( %d ) than valid and test." % config.get("nb_observed", t_train.shape[0]))
        t_train = t_train[:config.get("nb_observed", t_train.shape[0])]
        x_train = x_train[:, :config.get("nb_observed", t_train.shape[0])]
        train = data_utils.TensorDataset(x_train)
        dl_train = data_utils.DataLoader(train, batch_size=config.get("b_size", 100), shuffle=True, drop_last=True)

    # Creating the model
    starting_param = s.init_param
    fp = s.get_incomplete_physical_model(starting_param, trainable=True)

    config["lambda_p"] = config.get("lambda_0", float("nan"))

    trained_model, model, optimizer = get_models(solver, fp, device, config)
    #print(model)

    lambda_0 = config.get("lambda_0", float("nan"))
    tau_2 = config.get("tau_2", float("nan"))
    n_iter = config.get("n_iter", 1)
    all_x_train = x_train

    # Expert augmentation if there exist a trained model.
    if trained_model is not None and True:
        # Generating an augmented training set:
        all_x_train_augm = []
        all_zp_train_augm = []
        print("Generating augmented data...")
        for i in range(config.get("nb_augmentation", 1)):
            for j, x_train in enumerate(tqdm(dl_train)):
                x_train = x_train[0].to(device)
                with torch.no_grad():
                    x_train_augm, z_p_augm = trained_model.augmented_data(t_train, x_train)
                    all_x_train_augm.append(x_train_augm)
                    all_zp_train_augm.append(z_p_augm)
                    if config.get("combined_augmentation", True):
                        all_x_train_augm.append(x_train)
                        all_zp_train_augm.append(1/torch.zeros_like(z_p_augm))
        train = data_utils.TensorDataset(torch.cat(all_x_train_augm, 0), torch.cat(all_zp_train_augm, 0))
        dl_train = data_utils.DataLoader(train, batch_size=config.get("b_size", 100), shuffle=True, drop_last=True)

    # Variables to keep track of the loss along training and save the best model
    best_valid = float("inf")
    best_param = {}
    counter = 0
    valid_increase_nb = 0
    last_epochs_res = torch.zeros(5) # Only used for updating the lambda value of APHYNITY
    debug = False
    if not debug and config.get("normalize_loss", True):
        with torch.no_grad():
            baseline, _, _ = model.loss(t_train, x_train.to(device))
            baseline = baseline.item()
    else:
        print("Not normalizing the loss.")
        baseline = 1.

    for epoch in range(config["n_epochs"]):
        for i in range(n_iter):
            # Update step
            sum_loss = 0.
            sum_traj = 0.
            counter_loss = 0
            for x_train in tqdm(dl_train):

                if trained_model is not None and True:
                    x_train, z_p = x_train
                    x_train = x_train.to(device)
                    z_p = z_p.to(device)
                else:
                    x_train = x_train[0].to(device)
                if debug:
                    counter_loss = 1
                    fa_norm = torch.tensor(0.)
                    break
                loss, fa_norm, l_trajectory = model.loss(t_train, x_train)
                loss = loss / baseline
                if trained_model is not None and config.get("loss_params", True) and True:
                    est_param = model.predicted_parameters(t_train, x_train)
                    loss_param = ((est_param[~z_p.isinf()] - z_p[~z_p.isinf()]) ** 2).mean()
                    loss = loss + loss_param

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                sum_traj += l_trajectory
                counter_loss += 1

            loss_train = sum_loss / counter_loss
            mean_traj = sum_traj / counter_loss
            last_epochs_res[counter] = loss_train
            counter = (counter + 1) % 5

            with torch.no_grad():
                est_param_valid = model.predicted_parameters_as_dict(t_valid, x_valid, True)
                est_param_test_shifted = model.predicted_parameters_as_dict(t_test_shifted, x_test_shifted, True)
                est_param_train = model.predicted_parameters_as_dict(t_train, all_x_train, True)

                _, pred_x_valid = model.forward(t_valid, x_valid)
                _, pred_x_test_shifted = model.forward(t_test_shifted, x_test_shifted)
                _, pred_x_train = model.forward(t_train, all_x_train)

                loss_valid, fa_norm_valid, l_trajectory = model.loss(t_valid, x_valid)
                loss_test_shifted, fa_norm_test_shifted, l_trajectory = model.loss(t_test_shifted, x_test_shifted)
                if solver in ["APHYNITYDoublePendulum", "HybridVAEDoublePendulum"]:
                    se_train, _ = model.constraint_traj_from_sol(t_train, all_x_train, pred_x_train)
                    se_valid, _ = model.constraint_traj_from_sol(t_valid, x_valid, pred_x_valid)
                    se_valid = se_valid.unsqueeze(1)
                    se_test_shifted, _ = model.constraint_traj_from_sol(t_valid, x_test_shifted,
                                                                        pred_x_test_shifted)
                    se_test_shifted = se_test_shifted.unsqueeze(1)
                    print(se_valid.mean(), se_test_shifted.mean())
                else:
                    se_train = ((pred_x_train[:, model.nb_observed - 1:] -
                                 all_x_train[:, model.nb_observed - 1:]) ** 2).sum(4).sum(3).sum(2)
                    se_valid = ((pred_x_valid[:, model.nb_observed - 1:] -
                                 x_valid[:, model.nb_observed - 1:]) ** 2).sum(4).sum(3).sum(2)
                    se_test_shifted = ((pred_x_test_shifted[:, model.nb_observed - 1:] -
                                        x_test_shifted[:, model.nb_observed - 1:]) ** 2).sum(4).sum(3).sum(2)

                mu_log_mse_valid = se_valid.mean(1).log().mean()
                std_log_mse_valid = se_valid.mean(1).log().std()
                mse_train = se_train.mean()
                mse_valid = se_valid.mean()
                mse_test_shifted = se_test_shifted.mean()
                if verbose:
                    message = "Epoch {:d} - Training loss: {:4f} - Training error on trajectory: {:4f}" \
                              " - Validation loss: {:4f}" \
                              " - Test loss: {:4f}" \
                              " - Training log-mse: {:4f}" \
                              " - Validation log-mse: {:4f}" \
                              " - Test log-mse: {:4f}" \
                              " - Validation log-mse mu: {:4f} ± {:4f}" \
                              " - train |fa|: {:4f} - valid |fa|: {:4f}".format(epoch,
                                                                                loss_train,
                                                                                mean_traj,
                                                                                loss_valid.item(),
                                                                                loss_test_shifted.item(),
                                                                                mse_train.log().item(),
                                                                                mse_valid.log().item(),
                                                                                mse_test_shifted.log().item(),
                                                                                mu_log_mse_valid.item(),
                                                                                std_log_mse_valid.item(),
                                                                                fa_norm.item(),
                                                                                fa_norm_valid.item())

                    print(message)

                cur_valid = mse_valid.log()
                if best_valid > cur_valid:
                    fp = model.dec.fp if solver in ["APHYNITY"] else model.fp
                    pth = "%s/" % save_path
                    if solver in ["APHYNITY", "HybridVAE"]:
                        plot_curves_partial(t_train.cpu(),
                                            pred_x_train.cpu()[:, model.nb_observed - 1:, :2, 0, 0],
                                            all_x_train.squeeze(2).cpu(), message, fp.get_x_labels(), pth + "train_"
                                            + fp.get_name(), est_param_train, true_param_train)
                        plot_curves_partial(t_valid.cpu(),
                                            pred_x_valid.cpu()[:, model.nb_observed - 1:, :2, 0, 0],
                                            x_valid.squeeze(2).cpu(), message, fp.get_x_labels(), pth + "valid_"
                                            + fp.get_name(), est_param_valid, true_param_valid)
                        plot_curves_partial(t_test_shifted.cpu(),
                                            pred_x_test_shifted.cpu()[:, model.nb_observed - 1:, :2, 0, 0],
                                            x_test_shifted.squeeze(2).cpu(), message, fp.get_x_labels(),
                                            pth + "test_" + fp.get_name(),
                                            est_param_test_shifted, true_param_test_shifted)
                    elif solver in ["APHYNITYDoublePendulum", "HybridVAEDoublePendulum"]:
                        nb_observed_theta_0 = model.nb_observed_theta_0
                        nb_observed_theta_1 = model.nb_observed_theta_1
                        plot_curves_double_pendulum(t_train.cpu(),
                                            torch.sin(pred_x_train).cpu()[:, :, :2, 0, 0],
                                            torch.sin(all_x_train).cpu()[:, :, :2, 0, 0], message,
                                                    pth + "train" + fp.get_name(), nb_observed_theta_0,
                                                    nb_observed_theta_1, model.nb_observed,
                                                    est_param_train, true_param_train)
                        plot_curves_double_pendulum(t_valid.cpu(), torch.sin(pred_x_valid).cpu()[:, :, :2, 0, 0],
                                                    torch.sin(x_valid)[:, :, :2, 0, 0], message, pth + "valid_" +
                                                    fp.get_name(), nb_observed_theta_0,  nb_observed_theta_1,
                                                    model.nb_observed, est_param_valid, true_param_valid)
                        plot_curves_double_pendulum(t_test_shifted.cpu(),
                                            torch.sin(pred_x_test_shifted).cpu()[:, :, :2, 0, 0],
                                            torch.sin(x_test_shifted).cpu()[:, :, :2, 0, 0], message,
                                            pth + "test_" + "OOD_" + fp.get_name(), nb_observed_theta_0,
                                                    nb_observed_theta_1, model.nb_observed, est_param_test_shifted,
                                                    true_param_test_shifted)
                    if trained_model is not None:
                        save_name = solver + "_plus_best_valid_model.pt"
                    else:
                        save_name = solver + "_best_valid_model.pt"
                    if config.get("save_all_models", False):
                        torch.save(model.state_dict(), pth + save_name[:-3] + str(valid_increase_nb) + ".pt")
                        valid_increase_nb += 1
                    torch.save(model.state_dict(), pth + save_name)
                    if solver in ["APHYNITY", "APHYNITYReactionDiffusion",  "APHYNITYDoublePendulum"]:
                        for p in param_text:
                            best_param[p] = est_param_valid[p].squeeze(1)
                    best_valid = cur_valid.item()
                    best_test = mse_test_shifted.log().item()
                    print("New best validation log-mse at epoch: %d" % epoch)

        if "APHYNITY" in solver and (last_epochs_res.max() - last_epochs_res.min()).abs() / last_epochs_res.max().abs() < .2:
            print("Increase constraint weight")
            lambda_0 += tau_2 * model.constraint_traj(t_train, x_train)[0].mean().item()

        # Logging metrics.
        metrics = {'Progress': epoch,
                   'Train Loss': loss_train,
                   'Log-MSE Validation': mse_valid.log().item(),
                   'Validation Loss': loss_valid.item(),
                   'Log-MSE OOD': mse_test_shifted.log().item(),
                   'OOD Loss': loss_test_shifted.item(),
                   'Train |fa|': fa_norm.item(),
                   'Validation |fa|': fa_norm_valid.item(),
                   'Progress Text': message,
                   'best_valid': best_valid}


nb_exp = 1
sim_dic = {"DampedPendulum": DampedPendulum,
           "RLC": RLCCircuit,
           "ReactionDiffusion": ReactionDiffusion,
           "DoublePendulum": DoublePendulum}

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--config', type=str, default="code/scripts/configs/Pendulum/APHYNITY.yaml")
# Parse the argument
args = parser.parse_args()
config_name = args.config.split('/')[-1]
all_config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config = all_config["parameters"]
data_path = config.get("data_path", None)
now = datetime.now()
pth = "%s/runs/%s/%s" % (data_path, config_name, now.strftime("%m_%d_%Y_%H_%M_%S"))
if not os.path.exists(pth):
    os.makedirs(pth)
shutil.copyfile(args.config, pth + "/" + config_name)

s = sim_dic[config["simulator"]["name"]](**config["simulator"])

run_exp(s, True, config=config["optimization"], solver=config["optimization"].get("model", "None"), config_name=config_name,
        data_path=data_path, save_path=pth)
