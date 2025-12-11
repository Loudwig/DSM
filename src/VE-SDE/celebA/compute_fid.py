import os
import json
import math
from math import ceil

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

import models as m  


torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


def t(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)


# ------------------------
# Paths / experiment config
# ------------------------

BASE_PATH = os.getcwd()
RUNS_ROOT = os.path.join(BASE_PATH, "runs")

EXP_NAME = "comparaison"   # <- adapte si besoin
RUN_ID = "008"             # <- adapte si besoin

NUM_FID_SAMPLES = 10_000   # nombre d'images réelles / générées pour le FID
BATCH_SIZE_FID = 64

EXP_DIR = os.path.join(RUNS_ROOT, EXP_NAME)
RUN_DIR = os.path.join(EXP_DIR, RUN_ID)
WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
LOGS_DIR = os.path.join(RUN_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

with open(os.path.join(LOGS_DIR, "hparams.json"), "r") as f:
    hparams = json.load(f)

print("Loaded hparams from:", RUN_DIR)
print("sigma hyperparams:", hparams["sigma"])
print("img_size:", hparams["img_size"])

sigma_min = hparams["sigma"]["min"]
sigma_max = hparams["sigma"]["max"]

img_size = hparams["img_size"]
C = hparams["model"]["in_channel"]
H = W = img_size


state_dict = torch.load(
    os.path.join(WEIGHTS_DIR, "model_ema.pt"),
    map_location=device,
    weights_only=True,
)

model = m.SmallUNetSigma(
    in_ch=hparams["model"]["in_channel"],
    base_ch=hparams["model"]["base_ch"],
    channel_mults=hparams["model"]["channel_mults"],
    emb_dim=hparams["model"]["sigma_emb_dim"],
).to(device)

model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params/1e6:.2f}M parameters")


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self._log_ratio = math.log(self.sigma_max) - math.log(self.sigma_min)

    def sigma(self, t):
        # t \in [0,1]
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def g(self, t):
        # diffusion coeff g(t) s.t. dx = g(t) dW
        return torch.sqrt(torch.tensor(2.0 * self._log_ratio, device=t.device)) * self.sigma(t)

    def prior_sampling(self, shape, device):
        # p(x_T) ~ N(0, sigma_max^2 I)
        return torch.randn(*shape, device=device) * self.sigma_max


sde = VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=1000)


# ------------------------
# Corrector + PC sampler
# ------------------------

def langevin_corrector_snr(
    x,
    t_vec,
    score_model,
    snr=0.16,
    n_steps=1,
    sigma_cutoff=sigma_max,
    max_step_scale=np.inf,
):
    # Langevin corrector with target SNR (Song et al.)
    B = x.shape[0]

    for _ in range(n_steps):
        sigma_vec = sde.sigma(t_vec)          # (B,)
        sigma_batch = sigma_vec.view(B, 1)    # (B,1) for model

        # Only correct where sigma is not too large
        mask = (sigma_vec <= sigma_cutoff)
        if not mask.any():
            return x

        x_sub   = x[mask]
        sig_sub = sigma_batch[mask]

        grad  = score_model(x_sub, sig_sub)   # score(x, sigma(t))
        noise = torch.randn_like(x_sub)

        grad_norm  = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

        raw_step = 2.0 * (snr * noise_norm / (grad_norm + 1e-12))**2

        sigma2_mean = (sig_sub.squeeze(1)**2).mean()
        max_step = max_step_scale * sigma2_mean

        step_size = torch.clamp(raw_step, max=max_step)
        step_size_t = step_size.view(1, 1, 1, 1)

        x_mean_sub = x_sub + step_size_t * grad
        x_sub = x_mean_sub + torch.sqrt(2.0 * step_size_t) * noise

        x = x.clone()
        x[mask] = x_sub

    return x


@torch.no_grad()
def pc_sampler_ve_snr(
    num_samples=8,
    num_steps=1000,
    n_corrector_steps=1,
    snr=0.16,
    eps=1e-5,
):
    # Predictor-Corrector sampler (VE SDE + SNR-based Langevin corrector)
    model.eval()
    B = num_samples

    C = 3
    H = W = img_size
    x = sde.prior_sampling((B, C, H, W), device=device)

    t_grid = torch.linspace(1.0, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        t      = t_grid[i].expand(B)      # current time
        t_next = t_grid[i + 1].expand(B)  # next (smaller) time
        dt = (t_next[0] - t[0]).item()    # < 0

        g_t   = sde.g(t)                  # (B,)
        sigma_vec   = sde.sigma(t)        # (B,)
        sigma_batch = sigma_vec.view(B, 1)

        score = model(x, sigma_batch)     # (B,C,H,W)

        drift = -(g_t ** 2).view(B, 1, 1, 1) * score
        noise = torch.randn_like(x)

        x_mean = x + drift * dt
        x = x_mean + noise * torch.sqrt(torch.tensor(-dt, device=device)) \
                    * g_t.view(B, 1, 1, 1)

        # Corrector step at t_next
        x = langevin_corrector_snr(
            x, t_next, model,
            snr=snr,
            n_steps=n_corrector_steps,
        )

    return x


# Root of CelebA (must contain the images as an ImageFolder, e.g. data/celeba/img_align_celeba)
CELEBA_ROOT = os.path.join(BASE_PATH, "data", "celeba")
print("FID: CelebA root =", CELEBA_ROOT)

transform_fid = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(img_size),
    transforms.ToTensor(),          # -> [0,1]
])

real_dataset = datasets.ImageFolder(
    root=CELEBA_ROOT,
    transform=transform_fid,
)
print("FID: number of real images =", len(real_dataset))

real_loader = DataLoader(
    real_dataset,
    batch_size=BATCH_SIZE_FID,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

fid = FrechetInceptionDistance(
    feature=2048,
    normalize=True,   # expects floats in [0,1]
).to(device)

# ---- accumulate REAL features ----
num_real = min(NUM_FID_SAMPLES, len(real_dataset))
real_seen = 0

for real_batch, _ in real_loader:
    real_batch = real_batch.to(device)   # (B,3,H,W) in [0,1]
    fid.update(real_batch, real=True)
    real_seen += real_batch.size(0)
    if real_seen >= num_real:
        break

print(f"FID: accumulated {real_seen} real images.")


# ---- accumulate FAKE features ----
num_fake = num_real
num_batches_fake = ceil(num_fake / BATCH_SIZE_FID)

model.eval()
fake_seen = 0

for _ in range(num_batches_fake):
    curr_bs = min(BATCH_SIZE_FID, num_fake - fake_seen)
    if curr_bs <= 0:
        break

    x_fake = pc_sampler_ve_snr(
        num_samples=curr_bs,
        num_steps=1000,          # adapte si tu changes ton sampler "best"
        n_corrector_steps=1,
        snr=0.16,
        eps=1e-5,
    ).to(device)                 # (B,3,H,W) in [-1,1]

    # Invert training normalization: [-1,1] -> [0,1]
    x_fake_01 = (x_fake + 1.0) / 2.0
    x_fake_01 = x_fake_01.clamp(0.0, 1.0)

    fid.update(x_fake_01, real=False)
    fake_seen += curr_bs
    if fake_seen >= num_fake:
        break

print(f"FID: accumulated {fake_seen} generated images.")

fid_value = fid.compute().item()
print(f"FID (PC-SNR, {num_real} real / {num_fake} fake) = {fid_value:.4f}")

fid_path = os.path.join(LOGS_DIR, "fid_pc_snr.npy")
np.save(fid_path, np.array([fid_value], dtype=np.float32))
print(f"FID saved to {fid_path}")
