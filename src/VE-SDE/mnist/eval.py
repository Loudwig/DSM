# %%
import torch
import models as m
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# %%
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

# %%
# --------- Paths / hparams ---------
BASE_PATH = os.getcwd()
RUNS_ROOT = os.path.join(BASE_PATH, "runs")

EXP_NAME = "mnist_baseline"   # <-- adapte à ton exp
RUN_ID = "001"                # <-- adapte à ton run

EXP_DIR = os.path.join(RUNS_ROOT, EXP_NAME)
RUN_DIR = os.path.join(EXP_DIR, RUN_ID)
WEIGHTS_DIR = os.path.join(RUN_DIR, "weights")
LOGS_DIR = os.path.join(RUN_DIR, "logs")
SAMPLES_DIR = os.path.join(RUN_DIR, "samples_pc")
os.makedirs(SAMPLES_DIR, exist_ok=True)

with open(os.path.join(LOGS_DIR, "hparams.json"), "r") as f:
    hparams = json.load(f)

print("Loaded hparams from:", RUN_DIR)
print("sigma hyperparams:", hparams["sigma"])
print("dataset:", hparams.get("dataset", "unknown"))

sigma_min = hparams["sigma"]["min"]
sigma_max = hparams["sigma"]["max"]

# MNIST: 1x28x28
C = hparams["model"]["in_channel"]
H = W = 28

# %%
# --------- Load model ---------
state_dict = torch.load(
    os.path.join(WEIGHTS_DIR, "model.pt"),
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

# %%
# --------- VE schedule σ(t) and g(t) ---------
# Training: log σ(t) = log σ_min + t (log σ_max - log σ_min), t ∈ [0,1]

log_sigma_min = t(sigma_min).log()
log_sigma_max = t(sigma_max).log()
log_r = log_sigma_max - log_sigma_min   # log(σ_max / σ_min)

def sigma_of_t(t_scalar):
    """
    t_scalar: scalaire ou tensor in [0,1]
    returns σ(t) avec même dtype/device.
    """
    return torch.exp(log_sigma_min + t_scalar * log_r)

def g_of_t(t_scalar):
    """
    g(t) = sqrt(d/dt σ^2(t))
         = sqrt(2 log(σ_max/σ_min)) * σ(t)
    car σ^2(t) = σ_min^2 * (σ_max/σ_min)^{2t}.
    """
    sigma_t = sigma_of_t(t_scalar)
    return torch.sqrt(2.0 * log_r) * sigma_t

# Sanity check
for val in [0.0, 0.5, 1.0]:
    s = sigma_of_t(t(val))
    g = g_of_t(t(val))
    print(f"t={val:.2f} -> sigma={s.item():.4f}, g={g.item():.4f}")

# %%
# --------- Predictor: reverse VE SDE (Euler–Maruyama) ---------
@torch.no_grad()
def predictor_step(x, t_k, t_prev):
    """
    Reverse VE SDE:
      dx = -g(t)^2 s_theta(x,t) dt + g(t) dW_t, t: 1 -> 0.

    Discrétisation t_k -> t_prev < t_k:
      x_{k-1} = x_k
        - g(t_k)^2 s_theta(x_k,t_k) * |Δt|
        + g(t_k) sqrt(|Δt|) z_k,
      avec Δt = t_prev - t_k < 0.
    """
    dt = t_prev - t_k  # négatif
    assert (dt <= 0).all()

    sigma_k = sigma_of_t(t_k)  # scalaire
    g_k = g_of_t(t_k)          # scalaire

    B = x.shape[0]
    sigma_batch = sigma_k.view(1, 1).expand(B, 1)  # (B,1)
    score = model(x, sigma_batch)                  # (B,C,H,W)

    noise = torch.randn_like(x)
    x = x + (-g_k**2 * score) * dt + g_k * torch.sqrt(-dt) * noise
    return x

# %%
# --------- Corrector: simple Langevin at fixed t ---------
@torch.no_grad()
def corrector_step(x, t_k, n_steps, base_eps=1e-4):
    """
    Correcteur Langevin classique à temps fixé t_k :
      x^{(m+1)} = x^{(m)} + ε_k s_theta(x^{(m)}, t_k) + sqrt(2 ε_k) z^{(m)}

    ε_k = base_eps * σ(t_k)^2
    """
    if n_steps <= 0:
        return x

    sigma_k = sigma_of_t(t_k)
    eps_k = base_eps * sigma_k**2

    for _ in range(n_steps):
        B = x.shape[0]
        sigma_batch = sigma_k.view(1, 1).expand(B, 1)  # (B,1)
        score = model(x, sigma_batch)

        noise = torch.randn_like(x)
        x = x + eps_k * score + torch.sqrt(2.0 * eps_k) * noise

    return x

# %%
# --------- Predictor–Corrector sampler with snapshots ---------
@torch.no_grad()
def pc_sampler_ve(
    num_samples=4,
    num_steps=1000,
    n_corrector_steps=1,
    corrector_base_eps=1e-4,
):
    """
    Sampler VE SDE avec Predictor–Corrector :
    - t_0 = 0 < ... < t_N = 1, on itère de N -> 0.
    - Predictor: Euler–Maruyama sur la reverse SDE.
    - Corrector: Langevin avec ε_k = base_eps * σ(t_k)^2.
    - Prior: x_T ~ N(0, σ_max^2 I).

    On renvoie aussi des snapshots (premier, milieu, dernier, etc.).
    """

    # grille de temps
    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    t_grid_rev = t_grid.flip(0)  # [1, ..., 0]

    # prior p_T = N(0, σ_max^2 I)
    x = torch.randn(num_samples, C, H, W, device=device) * sigma_max
    print(f"Init shape: {x.shape}, prior σ_max={sigma_max}")

    # indices de snapshots: début, 25%, 50%, 75%, fin
    snapshot_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    snapshot_indices = sorted(
        {int(frac * num_steps) for frac in snapshot_fracs}
    )  # unique + trié
    print("Snapshot steps:", snapshot_indices)

    snapshots = {}  # step -> tensor (num_samples, C, H, W)

    # snapshot initial (avant toute update)
    snapshots[0] = x.clone()

    for i in range(num_steps):
        t_k = t_grid_rev[i]      # temps courant
        t_prev = t_grid_rev[i+1] # temps suivant (plus proche de 0)

        # predictor
        x = predictor_step(x, t_k, t_prev)

        # corrector
        x = corrector_step(x, t_k, n_corrector_steps, base_eps=corrector_base_eps)

        # sauvegarde snapshot si index match
        step_id = i + 1
        if step_id in snapshot_indices:
            snapshots[step_id] = x.clone()

        if (i + 1) % max(1, num_steps // 10) == 0:
            print(f"[{i+1}/{num_steps}] t={t_k.item():.4f}")

    # clamp pour affichage
    x_clamped = x.clamp(-1.0, 1.0)
    x_vis = (x_clamped + 1.0) / 2.0

    # on convertit aussi les snapshots en version [0,1] pour plotting
    snapshots_vis = {}
    for k, x_snap in snapshots.items():
        x_snap_clamped = x_snap.clamp(-1.0, 1.0)
        snapshots_vis[k] = (x_snap_clamped + 1.0) / 2.0

    return x, x_vis, snapshots_vis

# %%
# --------- Run sampler and visualize snapshots ---------
num_samples = 8
num_steps = 1000
n_corrector_steps = 1        # 1–2 suffit souvent
corrector_base_eps = 1e-4    # à tuner

x_final, x_vis, snapshots_vis = pc_sampler_ve(
    num_samples=num_samples,
    num_steps=num_steps,
    n_corrector_steps=n_corrector_steps,
    corrector_base_eps=corrector_base_eps,
)

# On ordonne les snapshots par étape
sorted_steps = sorted(snapshots_vis.keys())

# Figure multi-snapshots : chaque colonne = un temps, lignes = images
n_cols = len(sorted_steps)
n_rows = 1  # on affiche un grid par snapshot

plt.figure(figsize=(4 * n_cols, 4))
for col, step_id in enumerate(sorted_steps):
    x_snap_vis = snapshots_vis[step_id]
    grid = make_grid(
        x_snap_vis,
        nrow=int(np.sqrt(num_samples))
    )
    plt.subplot(1, n_cols, col + 1)
    plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap=None)
    plt.axis("off")
    frac = step_id / num_steps
    plt.title(f"step {step_id}/{num_steps}\n(t≈{1-frac:.2f})")

plt.suptitle(
    f"VE PC samples snapshots (MNIST)\nN={num_steps}, n_corr={n_corrector_steps}, eps0={corrector_base_eps}"
)
plt.tight_layout()
plt.show()

# Sauvegarde de la figure snapshots
snap_save_path = os.path.join(
    SAMPLES_DIR,
    f"snapshots_ve_pc_N{num_steps}_corr{n_corrector_steps}_eps{corrector_base_eps}.png",
)
plt.savefig(snap_save_path, dpi=200, bbox_inches="tight")
print("Saved snapshots figure to:", snap_save_path)

# Et on sauvegarde aussi uniquement le grid final si tu veux
final_grid = make_grid(x_vis, nrow=int(np.sqrt(num_samples)))
final_save_path = os.path.join(
    SAMPLES_DIR,
    f"samples_ve_pc_final_N{num_steps}_corr{n_corrector_steps}_eps{corrector_base_eps}.png",
)
plt.imsave(final_save_path, final_grid.permute(1, 2, 0).detach().cpu().numpy())
print("Saved final samples to:", final_save_path)
