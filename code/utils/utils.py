#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch
from matplotlib import pyplot as plt
from torch import nn

from code.hybrid_models import HybridAutoencoder
from code.hybrid_models.APHYNITY import APHYNITYAutoencoderDoublePendulum
from code.hybrid_models import APHYNITYAutoencoder, APHYNITYAutoencoderReactionDiffusion, HybridVAEDoublePendulum, \
    HybridVAEReactionDiffusion, HybridVAE


def compute_zp_metrics(true_zp, est_zp, prefix, device, list_param):
    sum_cur_rel_errors = 0
    metrics = {}
    for p in list_param:
        if p in true_zp and p in est_zp:
            true_zp_c = true_zp[p].to(device)
            est_zp_c = est_zp[p].squeeze(1).to(device)
            metrics[prefix + "MAE {}".format(p)] = (true_zp_c - est_zp_c).abs().mean().item()
            metrics[prefix + "cur_rel_error_" + p] = ((est_zp_c - true_zp_c).abs() / true_zp_c).mean().item() * 100
            sum_cur_rel_errors += metrics[prefix + "cur_rel_error_" + p]

    metrics[prefix + "avg_cur_rel_errors_"] = sum_cur_rel_errors / len(list_param)

    return metrics


def get_models(solver, fp, device, config) -> tuple[HybridAutoencoder, HybridAutoencoder, torch.optim.Optimizer]:
    if solver in ["APHYNITY"]:
        trained_model = APHYNITYAutoencoder(fp.to(device), device=device, **config).to(device)
        model = APHYNITYAutoencoder(fp.to(device), device=device, **config).to(device)
    elif solver in ["APHYNITYReactionDiffusion"]:
        trained_model = APHYNITYAutoencoderReactionDiffusion(fp.to(device), device=device, **config).to(device)
        model = APHYNITYAutoencoderReactionDiffusion(fp.to(device), device=device, **config).to(device)
    elif solver in ["APHYNITYDoublePendulum"]:
        trained_model = APHYNITYAutoencoderDoublePendulum(fp.to(device), device=device, **config).to(device)
        model = APHYNITYAutoencoderDoublePendulum(fp.to(device), device=device, **config).to(device)
    elif config["model"] == "HybridVAE":
        trained_model = HybridVAE(fp.to(device), device=device, **config).to(device)
        model = HybridVAE(fp.to(device), device=device, **config).to(device)
    elif config["model"] == "HybridVAEReactionDiffusion":
        trained_model = HybridVAEReactionDiffusion(fp.to(device), device=device, **config).to(device)
        model = HybridVAEReactionDiffusion(fp.to(device), device=device, **config).to(device)
    elif config["model"] == "HybridVAEDoublePendulum":
        trained_model = HybridVAEDoublePendulum(fp.to(device), device=device, **config).to(device)
        model = HybridVAEDoublePendulum(fp.to(device), device=device, **config).to(device)
    else:
        raise Exception("The model chosen does not exist.")

    if "path_model" in config and config["augmented"]:
        if config["model"] == "HybridVAE":
            to_train = nn.ModuleList([model.ga, model.gp_1, model.gp_2])
        elif config["model"] == "HybridVAEReactionDiffusion":
            to_train = nn.ModuleList([model.enc_za, model.enc_zp, model.gp_1])
        elif config["model"] == "HybridVAEDoublePendulum":
            to_train = nn.ModuleList([model])
        else:
            to_train = model.enc

        print("Loading models...")
        trained_model.load_state_dict(torch.load(config["path_model"], map_location=device))
        model.load_state_dict(torch.load(config["path_model"], map_location=device))

        if config["model"] in ["HybridVAE"]:
            model.sigma_x = torch.nn.Parameter(torch.zeros_like(model.sigma_x))
        elif config["model"] in ["HybridVAEDoublePendulum"]:
            model.sigma_x_cos = nn.Parameter(torch.zeros(2, requires_grad=True)).to(device)
            model.sigma_x_sin = nn.Parameter(torch.zeros(2, requires_grad=True)).to(device)

        optimizer = torch.optim.Adam(to_train.parameters(),
                                     lr=config["learning_rate_fa"],
                                     weight_decay=config["weight_decay_fa"])
    else:
        trained_model = None
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate_fa"],
                                     weight_decay=config["weight_decay_fa"])

    return trained_model, model, optimizer


def plot_augmented_datasets(all_zp_train_augm, save_path):
    plt.figure(figsize=(15, 20))
    plt.subplot(2, 2, 1)
    plt.xlim(xmin=-4, xmax=4)
    plt.title("$ \\theta_0$")
    z_p = torch.cat(all_zp_train_augm, 0).permute(1, 0)[[0]]
    z_p = z_p[~z_p.isinf()]
    plt.hist(z_p, bins=40, histtype="stepfilled", alpha=.3, density=True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("$ \\theta_1$")
    plt.xlim(xmin=-4, xmax=4)
    z_p = torch.cat(all_zp_train_augm, 0).permute(1, 0)[[1]]
    z_p = z_p[~z_p.isinf()]
    plt.hist(z_p, bins=40, histtype="stepfilled", alpha=.3,
             density=True)
    plt.subplot(2, 2, 3)
    plt.xlim(xmin=-25, xmax=25)
    plt.title("$\\dot \\theta_0$")
    z_p = torch.cat(all_zp_train_augm, 0).permute(1, 0)[[2]]
    z_p = z_p[~z_p.isinf()]
    plt.hist(z_p, bins=40, histtype="stepfilled", alpha=.3,
             density=True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("$\\dot \\theta_1$")
    plt.xlim(xmin=-50, xmax=70)
    z_p = torch.cat(all_zp_train_augm, 0).permute(1, 0)[[3]]
    z_p = z_p[~z_p.isinf()]
    plt.hist(z_p, bins=40, histtype="stepfilled", alpha=.3,
             density=True)
    plt.savefig("%s/init_state_dist.png" % save_path)

