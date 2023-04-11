#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import matplotlib.pyplot as plt
from textwrap import wrap

import torch


def plot_curves(t, x_pred, x_obs, title, labels, save_name):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("\n".join(wrap(title, 150)), fontsize=18)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(t, x_pred[i], '--', linewidth=2.)
        for j in range(x_obs.shape[2]):
            plt.scatter(t, x_obs[i, :, j], marker='x', linewidth=1.)
        if i == 0:
            plt.legend(["Predicted %s" % labels[0], "Predicted %s" % labels[1],
                        "Realized %s" % labels[0], "Realized %s" % labels[1]])
            plt.xlabel("Time - t")
            plt.ylabel("State - X(t)")
    plt.savefig(save_name + ".pdf")
    print("Figure saved in %s" % save_name)
    plt.close(fig)


def plot_curves_partial(t, x_pred, x_obs, title, labels, save_name, est_param=None, true_param=None):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("\n".join(wrap(title, 150)), fontsize=18)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(t[-x_pred.shape[1]:], x_pred[i], '--', linewidth=2.)
        for j in range(x_obs.shape[2]):
            plt.scatter(t, x_obs[i, :, j], marker='x', linewidth=1.)
        if i == 0:
            plt.legend(["Predicted %s" % labels[0], "Predicted %s" % labels[1],
                        "Realized %s" % labels[0], "Realized %s" % labels[1]])
            plt.xlabel("Time - t")
            plt.ylabel("State - X(t)")
        if est_param is not None:
            message = "".join(["{}: True: {:.4f} Est: {:.4f} -- ".format(p,
                                                                         true_param[p][i],
                                                                         est_param[p][i, 0]) for p in est_param.keys()])
        else:
            message = "Sin"
        plt.title("\n".join(wrap(message[:-3], 60)), fontsize=13)
    plt.tight_layout()
    plt.savefig(save_name + ".pdf")
    print("Figure saved in %s" % save_name)
    plt.close(fig)


def plot_curves_complete(t, x_pred, x_obs, title, labels, save_name, nb_observed=-1, est_param=None, true_param=None):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("\n".join(wrap(title, 150)), fontsize=18)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(t[-x_pred.shape[1]:], x_pred[i], '--', linewidth=2.)
        if nb_observed > 0:
            plt.vlines(t[nb_observed], x_obs.min(), x_obs.max(), label='Nb Observed')
        for j in range(x_obs.shape[2]):
            plt.scatter(t, x_obs[i, :, j], marker='x', linewidth=1.)
        if i == 0:
            plt.legend(["Predicted %s" % labels[0], "Predicted %s" % labels[1],
                        "Realized %s" % labels[0], "Realized %s" % labels[1]])
            plt.xlabel("Time - t")
            plt.ylabel("State - X(t)")
        if est_param is not None:
            message = "".join(["${}$: True: {:.4f} Est: {:.4f} \n".format(p,
                                                                         true_param[p][i],
                                                                         est_param[p][i, 0]) for p in est_param.keys()])
        else:
            message = "Sin"
        plt.title(message[:-3], fontsize=13)
        #plt.title("\n".join(wrap(message[:-3], 60)), fontsize=13)
    plt.tight_layout()
    plt.savefig(save_name + ".pdf")
    print("Figure saved in %s" % save_name)
    plt.close(fig)


def plot_curves_double_pendulum(t, x_pred, x_obs, title, save_name, nb_observed_theta_0, nb_observed_theta_1,
                                nb_observed, est_param=None, true_param=None):
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle("\n".join(wrap(title, 150)), fontsize=18)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot(t, x_pred[i], '--', linewidth=2.)
        plt.scatter(t[nb_observed-nb_observed_theta_0:nb_observed],
                    x_obs[i, nb_observed-nb_observed_theta_0:nb_observed, 0], marker='o', linewidth=1., c='blue')
        plt.scatter(t[nb_observed-nb_observed_theta_1:nb_observed],
                    x_obs[i, nb_observed-nb_observed_theta_1:nb_observed, 1], marker='o', linewidth=1, c='orange')

        if nb_observed - nb_observed_theta_0 > 0:
            time_theta0 = torch.cat((t[:nb_observed-nb_observed_theta_0], t[nb_observed:]), 0)
            real_theta0 = torch.cat((x_obs[i, :nb_observed-nb_observed_theta_0, 0], x_obs[i, nb_observed:, 0]), 0)
        else:
            time_theta0 = t[nb_observed:]
            real_theta0 = x_obs[i, nb_observed:, 0]
        plt.scatter(time_theta0, real_theta0, marker='x', linewidth=1., c='blue')

        if nb_observed - nb_observed_theta_1 > 0:
            time_theta1 = torch.cat((t[:nb_observed-nb_observed_theta_1], t[nb_observed:]), 0)
            real_theta1 = torch.cat((x_obs[i, :nb_observed-nb_observed_theta_1, 1], x_obs[i, nb_observed:, 1]), 0)
        else:
            time_theta1 = t[nb_observed:]
            real_theta1 = x_obs[i, nb_observed:, 1]
        plt.scatter(time_theta1, real_theta1, marker='x', linewidth=1., c='orange')

        if i == 0:
            plt.legend(["Predicted $\\theta\_0$", "Predicted $\\theta\_1$",
                        "Observed $\\theta\_0$", "Observed $\\theta\_1$",
                        "Realized $\\theta\_0$", "Realized $\\theta\_1$"])
            plt.xlabel("Time - t")
            plt.ylabel("State - X(t)")
        if est_param is not None:
            message = "".join(["${}$: True: {:.4f} Est: {:.4f} \n".format(p,
                                                                         true_param[p][i],
                                                                         est_param[p][i, 0]) for p in est_param.keys()])
        else:
            message = "Sin"
        plt.title("\n".join(wrap(message[:-3], 90)), fontsize=13)
        #plt.title("\n".join(wrap("Sin", 60)), fontsize=13)
    plt.tight_layout()
    plt.savefig(save_name + ".pdf")
    print("Figure saved in %s" % save_name)
    plt.close(fig)


def plot_diffusion(x_pred, x_obs, title, labels, save_name, est_param, true_param):
    fig = plt.figure(figsize=(24, 30))
    fig.suptitle("\n".join(wrap(title, 250)), fontsize=24)
    nb_sec = 5
    for i in range(nb_sec):
        plt.subplot(6, nb_sec, i + 1)
        plt.imshow(x_pred[0, i * 10, 0], cmap='BrBG')
        plt.subplot(6, nb_sec, nb_sec + i + 1)
        plt.imshow(x_obs[0, i * 10, 0], cmap='BrBG')
        plt.subplot(6, nb_sec, 2*nb_sec + i + 1)
        plt.imshow(x_obs[0, i * 10, 0] - x_pred[0, i * 10, 0], cmap='BrBG')

        plt.subplot(6, nb_sec, nb_sec*3 + i + 1)
        plt.imshow(x_pred[0, i * 10, 1], cmap='PiYG')
        plt.subplot(6, nb_sec, nb_sec*4 + i + 1)
        plt.imshow(x_obs[0, i * 10, 1], cmap='PiYG')
        plt.subplot(6, nb_sec, nb_sec*5 + i + 1)
        plt.imshow(x_obs[0, i * 10, 1] - x_pred[0, i * 10, 1], cmap='PiYG')
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig(save_name + ".pdf")
    print("Figure saved in %s" % save_name)
    plt.close(fig)