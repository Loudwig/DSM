#!/usr/bin/env python

import argparse
import os
import json
import shutil

import matplotlib
matplotlib.use("Agg")  # for headless HPC
import matplotlib.pyplot as plt

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import models as m  # your SmallUNetSigma etc.

# -------------------------
# Global config
# -------------------------

CELEBA_ROOT = "data/celeba"
device = torch.device("cpu")


# -------------------------
# Utils
# -------------------------

def LossNCSN(model, x, sigmas, return_per_sigma=False):
    B = x.size(0)
    device_x = x.device
    n_sigmas = len(sigmas)

    idx = torch.randint(0, n_sigmas, (B,), device=device_x)
    sigma_vals = sigmas[idx]  # (B,)

    sigma_noise = sigma_vals.view(B, 1, 1, 1)

    eps = torch.randn_like(x)
    x_noisy = x + sigma_noise * eps

    s_hat = model(x_noisy, sigma_vals.view(B, 1))
    target = -eps / (sigma_noise + 1e-8)
    residual = (s_hat - target) ** 2

    w = (sigma_vals ** 2).view(B, 1, 1, 1)

    # loss par sample (B,)
    loss_per_sample = (w * residual).view(B, -1).mean(dim=1)
    loss = loss_per_sample.mean()

    if not return_per_sigma:
        return loss

    # somme des losses par niveau + compteur par niveau (detach -> pas de grad)
    loss_sum = torch.zeros(n_sigmas, device=device_x)
    count_sum = torch.zeros(n_sigmas, device=device_x)

    loss_sum.index_add_(0, idx, loss_per_sample.detach())
    count_sum.index_add_(0, idx, torch.ones_like(loss_per_sample))

    return loss, loss_sum.cpu(), count_sum.cpu()


def t(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype()).to(device)


def construct_noise_linspace(min_val, max_val, L):
    return torch.linspace(min_val, max_val, L).flip(0).to(device)


def construct_noise_logspace(sigma_min, sigma_max, L):
    return torch.logspace(
        torch.log10(t(sigma_min)),
        torch.log10(t(sigma_max)), L
    ).flip(0).to(device)


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


# -------------------------
# Argparse
# -------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Score-based model on CelebA (train only)"
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default="debugging")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)

    # NEW: gradient clipping
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="max_norm for clip_grad_norm_. Set 0 to disable.")

    # Sigma hyperparameters
    parser.add_argument("--sigma-min", type=float, default=1e-1)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--n-sigmas", type=int, default=10)
    parser.add_argument("--sigma-schedule", type=str, default="lin",
                        choices=["lin", "log"])

    # Model hyperparameters
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4")

    return parser.parse_args()


# -------------------------
# Main
# -------------------------

def main(args):
    global device

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.version())
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))

    torch.set_default_dtype(torch.float32)

    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    SAVE = not args.no_save

    def get_next_run_id(exp_dir: str) -> str:
        existing = [
            d for d in os.listdir(exp_dir)
            if d.isdigit() and len(d) == 3
        ]
        if not existing:
            return "001"
        return f"{int(max(existing)) + 1:03d}"

    BASE_PATH = os.getcwd()
    RUNS_ROOT = os.path.join(BASE_PATH, "runs")
    os.makedirs(RUNS_ROOT, exist_ok=True)

    EXP_NAME = args.exp_name
    EXP_DIR = os.path.join(RUNS_ROOT, EXP_NAME)
    os.makedirs(EXP_DIR, exist_ok=True)

    RUN_ID = get_next_run_id(EXP_DIR)
    RUN_DIR = os.path.join(EXP_DIR, RUN_ID)
    print("RUN_ID:", RUN_ID)

    if SAVE:
        os.makedirs(RUN_DIR, exist_ok=True)
        FIG_DIR = os.path.join(RUN_DIR, "figures")
        WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
        LOGS_DIR = os.path.join(RUN_DIR, "logs")
        os.makedirs(FIG_DIR, exist_ok=True)
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
    else:
        FIG_DIR = WEIGHTS_DIR = LOGS_DIR = None

    # Move SLURM out/err into RUN_DIR
    job_name = os.environ.get("SLURM_JOB_NAME")
    job_id = os.environ.get("SLURM_JOB_ID")
    if SAVE and job_name and job_id:
        out_src = os.path.join(RUNS_ROOT, f"{job_name}_{job_id}.out")
        err_src = os.path.join(RUNS_ROOT, f"{job_name}_{job_id}.err")
        out_dst = os.path.join(RUN_DIR, f"{job_name}_{job_id}.out")
        err_dst = os.path.join(RUN_DIR, f"{job_name}_{job_id}.err")

        for src, dst in [(out_src, out_dst), (err_src, err_dst)]:
            if os.path.exists(src):
                try:
                    shutil.move(src, dst)
                    print(f"Moved {src} -> {dst}")
                except Exception as e:
                    print(f"Could not move {src} to {dst}: {e}")

    # Dataset
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)
    ])

    full_data = datasets.ImageFolder(
        root=CELEBA_ROOT,
        transform=transform,
    )

    print("CelebA root:", CELEBA_ROOT)
    print("Nb d'images :", len(full_data))

    dataloader = DataLoader(
        full_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Training config
    batch_size = args.batch_size
    N_EPOCH = args.epochs
    EVAL_EVERY = args.eval_every
    lr = args.lr
    grad_clip = args.grad_clip

    sigma_min = args.sigma_min
    sigma_max = args.sigma_max
    n_sigmas = args.n_sigmas
    sigma_schedule = args.sigma_schedule

    if sigma_schedule == "log":
        sigmas = construct_noise_logspace(sigma_min, sigma_max, n_sigmas)
    elif sigma_schedule == "lin":
        sigmas = construct_noise_linspace(sigma_min, sigma_max, n_sigmas)
    else:
        raise ValueError(f"Unknown SIGMA_SCHEDULE: {sigma_schedule}")

    in_ch = 3
    base_ch = args.base_ch
    channel_mults = tuple(parse_int_list(args.channel_mults))
    SIGMA_EMB_DIM = 16

    print(f"Sigma schedule: {sigma_schedule}, "
          f"min={sigma_min}, max={sigma_max}, n={n_sigmas}")
    print(f"Model: base_ch={base_ch}, channel_mults={channel_mults}")
    print(f"Grad clip max_norm: {grad_clip}")

    model = m.SmallUNetSigma(
        in_ch=in_ch,
        base_ch=base_ch,
        channel_mults=channel_mults,
        emb_dim=SIGMA_EMB_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # si tu veux du weight decay: remplace par AdamW(..., weight_decay=1e-4)

    # Save hyperparams
    if SAVE:
        hparams = {
            "batch_size": batch_size,
            "N_epoch": N_EPOCH,
            "EVAL_EVERY": EVAL_EVERY,
            "lr": lr,
            "grad_clip": grad_clip,
            "sigma": {
                "schedule": sigma_schedule,
                "min": sigma_min,
                "max": sigma_max,
                "n_sigmas": n_sigmas,
                "values": [float(s) for s in sigmas],
            },
            "device": str(device),
            "model": {
                "in_channel": in_ch,
                "base_ch": base_ch,
                "channel_mults": list(channel_mults),
                "sigma_emb_dim": SIGMA_EMB_DIM,
            },
            "celeba_root": CELEBA_ROOT,
            "slurm": {
                "job_name": job_name,
                "job_id": job_id,
            },
        }

        with open(os.path.join(LOGS_DIR, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=4)

    total = sum(p.numel() for p in model.parameters())
    print(f"{total/1e6:.2f} M params")

    # -------------------------
    # Training loop
    # -------------------------
    model.train()
    L = []
    eval_steps = []
    step = 0

    # NEW: running loss per sigma (sur la fenêtre EVAL_EVERY)
    loss_sigma_running = torch.zeros(n_sigmas)
    count_sigma_running = torch.zeros(n_sigmas)
    L_sigma_history = []   # liste de vecteurs (n_sigmas,) à chaque eval

    for epoch in range(N_EPOCH):
        for u, (x, _) in enumerate(dataloader):
            step += 1

            x = x.to(device)
            optimizer.zero_grad()

            # NEW: on récupère somme+count par sigma
            loss, loss_sigma_sum, count_sigma_sum = LossNCSN(
                model, x, sigmas, return_per_sigma=True
            )

            loss.backward()

            # NEW: gradient clipping
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            L.append(loss.item())

            # accumulate per-sigma stats (CPU tensors)
            loss_sigma_running += loss_sigma_sum
            count_sigma_running += count_sigma_sum

            if (step) % EVAL_EVERY == 0:
                model.eval()
                eval_steps.append(step + 1)

                print(
                    f"[step : {step} | epoch : {epoch}] "
                    f"train loss (last {EVAL_EVERY}) = {np.mean(L[-EVAL_EVERY:]):.4f}"
                )

                # NEW: mean loss per sigma sur la fenêtre
                mean_sigma = (loss_sigma_running /
                              (count_sigma_running + 1e-8)).numpy()
                L_sigma_history.append(mean_sigma)

                # reset window accumulators
                loss_sigma_running.zero_()
                count_sigma_running.zero_()

                if SAVE:
                    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "model.pt"))
                    np.save(os.path.join(LOGS_DIR, "train_loss.npy"), np.array(L))
                    np.save(os.path.join(LOGS_DIR, "steps_eval.npy"), np.array(eval_steps))
                    np.save(os.path.join(LOGS_DIR, "train_loss_per_sigma.npy"),
                            np.array(L_sigma_history))
                    np.save(os.path.join(LOGS_DIR, "sigmas.npy"),
                            sigmas.detach().cpu().numpy())

                model.train()

    # Final save
    if SAVE:
        np.save(os.path.join(LOGS_DIR, "train_loss.npy"), np.array(L))
        np.save(os.path.join(LOGS_DIR, "steps_eval.npy"), np.array(eval_steps))
        np.save(os.path.join(LOGS_DIR, "train_loss_per_sigma.npy"),
                np.array(L_sigma_history))
        np.save(os.path.join(LOGS_DIR, "sigmas.npy"),
                sigmas.detach().cpu().numpy())
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "model.pt"))


if __name__ == "__main__":
    args = get_args()
    main(args)
