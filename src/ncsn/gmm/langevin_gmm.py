# %%
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import math

device = 'cpu'

def t(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype()).to(device)

print("Device:", device)

# %% [markdown]
# ## 1. GMM "difficile" + score exact

# %%
# GMM beaucoup plus contrasté :
# - 1 gros mode très large (poids élevé),
# - 3 modes moyens avec covariances standard,
# - 2 modes très rares et ultra-sharp,
# - 1 mode intermédiaire sharp.

means_list = [
    [-6.0,  0.0],   # k=0 (petit mode)
    [ 6.0,  0.0],   # k=1 (petit mode)
    [ 0.0,  6.0],   # k=2 (mode moyen)
    [ 0.0, -6.0],   # k=3 (mode moyen)
    [ 0.0,  0.0],   # k=4 (gros mode très large, dominant)
    [ 8.0,  8.0],   # k=5 (mode très rare, ultra-sharp)
    [-8.0, -8.0],   # k=6 (mode très rare, ultra-sharp)
]

mean = torch.stack([t(m) for m in means_list], dim=0).to(device)  # (K,2)
K = mean.size(0)

cov_list = [
    0.4 * torch.eye(2, device=device),   # k=0
    0.4 * torch.eye(2, device=device),   # k=1
    0.7 * torch.eye(2, device=device),   # k=2
    0.7 * torch.eye(2, device=device),   # k=3
    3.0 * torch.eye(2, device=device),   # k=4 (gros mode très large)
    0.05 * torch.eye(2, device=device),  # k=5 (ultra-sharp)
    0.05 * torch.eye(2, device=device),  # k=6 (ultra-sharp)
]

cov = torch.stack(cov_list, dim=0).to(device)  # (K,2,2)

# Poids très contrastés :
# - gros mode central = 0.45
# - modes moyens ~ 0.12-0.15
# - modes rares ultra-sharp = 0.02 chacun
weights_list = [0.08, 0.10, 0.12, 0.11, 0.45, 0.02, 0.12]  # somme ~ 1
w = t(weights_list)
w = w / w.sum()  # just in case
weights = dist.Categorical(w)

gaussians = dist.MultivariateNormal(mean, cov)
target_dist = dist.MixtureSameFamily(weights, gaussians)

print("K =", K)
print("Means:\n", mean)
print("Weights:\n", w)

# %%
def score(distrib, x):
    """
    Score exact ∇_x log p(x) via autograd.
    x: (N,2)
    """
    x = x.clone().detach().requires_grad_(True)
    lp = distrib.log_prob(x)
    s, = torch.autograd.grad(lp.sum(), x)
    return s

true_clean_score = lambda x: score(target_dist, x)


# %% [markdown]
# ## 2. Distributions bruitées p_σ et scores exacts p_σ

# %%
def construct_GMM_noised_distribution(noise_list, loc, cov, weights):
    d = loc.size(-1)
    noised = []
    for sigma in noise_list:
        g = dist.MultivariateNormal(
            loc,
            cov + torch.eye(d, device=device) * sigma**2
        )
        noised.append(dist.MixtureSameFamily(weights, g))
    return noised

SIGMA_MIN = 0.1
SIGMA_MAX = 20.0   # encore plus de "smoothing" au départ
N_SIGMAS  = 10

sigmas = torch.logspace(
    torch.log10(t(SIGMA_MIN)),
    torch.log10(t(SIGMA_MAX)),
    N_SIGMAS,
    device=device
).flip(0)

print("Sigmas (du gros bruit au petit):", sigmas)

true_noised_dist = construct_GMM_noised_distribution(
    sigmas, mean, cov, weights
)

true_noised_scores = [
    (lambda d: (lambda x: score(d, x)))(d)
    for d in true_noised_dist
]


# %% [markdown]
# ## 3. Norme du score *théorique* (true) pour calibrer SNR

# %%
def estimate_score_norm_true(distrib, score_fn, n_samples=30000):
    """
    E_{X~distrib}[||score_true(X)||²] via MC, mais avec le score exact (théorique).
    """
    x = distrib.sample((n_samples,)).to(device)
    s = score_fn(x)
    return (s**2).sum(dim=1).mean().item()

norm_clean_true = estimate_score_norm_true(target_dist, true_clean_score, n_samples=30000)
print("E[||score_clean||²] ≈", norm_clean_true)

norms_true_sigmas = []
for i in range(len(sigmas)):
    n_val = estimate_score_norm_true(true_noised_dist[i], true_noised_scores[i], n_samples=30000)
    norms_true_sigmas.append(n_val)
norms_true_sigmas = np.array(norms_true_sigmas, dtype=np.float64)
print("E[||score_σ||²] pour chaque σ:", norms_true_sigmas)


# %% [markdown]
# ## 4. Clustering et histogrammes

# %%
def assign_clusters(x, means):
    x = x.to(means.device)
    x_exp = x.unsqueeze(1)           # (N,1,2)
    m_exp = means.unsqueeze(0)       # (1,K,2)
    dist2 = ((x_exp - m_exp)**2).sum(dim=-1)   # (N,K)
    return dist2.argmin(dim=1)

def plot_cluster_hist(samples, means, true_weights, title):
    samples_cpu = samples.detach().cpu()
    means_cpu   = means.detach().cpu()
    true_w      = true_weights.detach().cpu().numpy()

    clusters = assign_clusters(samples_cpu, means_cpu)
    K = means_cpu.size(0)
    counts = torch.bincount(clusters, minlength=K).float()
    emp = (counts / counts.sum()).numpy()

    idx = np.arange(K)
    width = 0.35

    plt.figure(figsize=(7,4))
    plt.bar(idx - width/2, true_w, width, label="poids théoriques")
    plt.bar(idx + width/2, emp,    width, label="proportions finales")
    plt.xticks(idx, [f"k={i}" for i in range(K)])
    plt.ylim(0, 0.6)
    plt.ylabel("proportion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 5. Langevin simple avec score de p (un seul niveau de bruit)

# %%
def langevin_simple_true(
    target_dist,
    score_fn,
    prior_dist,
    SNR,
    norm_score_true,
    n_steps,
    n_chain
):
    D = 2
    tau = 2 * D * SNR / norm_score_true
    noise_std = math.sqrt(2 * tau)
    print(f"[Langevin simple] tau={tau:.3e}, noise_std={noise_std:.3e}")

    X = prior_dist.sample((n_chain,)).to(device)

    for k in range(n_steps):
        drift = score_fn(X)
        X = X + tau * drift + noise_std * torch.randn_like(X)

    return X


# %% [markdown]
# ## 6. Annealed Langevin (ALD) avec scores exacts p_σ

# %%
def annealed_langevin_true(
    prior,
    true_noisy_scores,
    sigmas,
    SNR,
    norms_true_sigmas,
    T_per_level,
    n_chain
):
    D = 2
    X = prior.sample((n_chain,)).to(device)

    for i in range(len(sigmas)):
        tau_i = 2 * D * SNR / norms_true_sigmas[i]
        noise_std_i = math.sqrt(2 * tau_i)
        print(f"[ALD] niveau {i}, σ={float(sigmas[i]):.3g}, "
              f"tau={tau_i:.3e}, noise_std={noise_std_i:.3e}")

        for _ in range(T_per_level):
            drift = true_noisy_scores[i](X)
            X = X + tau_i * drift + noise_std_i * torch.randn_like(X)

    return X


# %% [markdown]
# ## 7. Prior + hyperparamètres

# %%
ALD_prior_mean = [0.0, 0.0]
ALD_prior_cov_scalar = 10.0  # prior encore plus large pour bien voir comment ça se contracte
SNR = 0.1

prior_mean = t(ALD_prior_mean)
prior_cov  = ALD_prior_cov_scalar * torch.eye(2, device=device)
prior_dist = dist.MultivariateNormal(prior_mean, prior_cov)

N_CHAIN = 8000

ALD_T_PER_LEVEL = 10
L_simple_steps = ALD_T_PER_LEVEL * len(sigmas)


# %% [markdown]
# ## 8. Run des deux algos

# %%
X_langevin = langevin_simple_true(
    target_dist=target_dist,
    score_fn=true_clean_score,
    prior_dist=prior_dist,
    SNR=SNR,
    norm_score_true=norm_clean_true,
    n_steps=L_simple_steps,
    n_chain=N_CHAIN
).detach().cpu()

X_ald = annealed_langevin_true(
    prior=prior_dist,
    true_noisy_scores=true_noised_scores,
    sigmas=sigmas,
    SNR=SNR,
    norms_true_sigmas=norms_true_sigmas,
    T_per_level=ALD_T_PER_LEVEL,
    n_chain=N_CHAIN
).detach().cpu()

X_true = target_dist.sample((N_CHAIN,)).cpu()


# %% [markdown]
# ## 9. Scatter + histogrammes

# %%
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.scatter(X_true[:,0], X_true[:,1], s=4, alpha=0.5)
plt.scatter(mean[:,0].cpu(), mean[:,1].cpu(), c='red', marker='x', label="means")
plt.title("Samples vrais du GMM difficile")
plt.xlim(-12,12); plt.ylim(-12,12)
plt.gca().set_aspect('equal')
plt.legend()

plt.subplot(1,3,2)
plt.scatter(X_langevin[:,0], X_langevin[:,1], s=4, alpha=0.5)
plt.scatter(mean[:,0].cpu(), mean[:,1].cpu(), c='red', marker='x')
plt.title("Langevin simple (score p)")
plt.xlim(-12,12); plt.ylim(-12,12)
plt.gca().set_aspect('equal')

plt.subplot(1,3,3)
plt.scatter(X_ald[:,0], X_ald[:,1], s=4, alpha=0.5)
plt.scatter(mean[:,0].cpu(), mean[:,1].cpu(), c='red', marker='x')
plt.title("Annealed Langevin (scores p_σ)")
plt.xlim(-12,12); plt.ylim(-12,12)
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()

# %%
plot_cluster_hist(
    X_langevin,
    mean,
    w,
    title="Proportions finales – Langevin simple (GMM difficile)"
)

plot_cluster_hist(
    X_ald,
    mean,
    w,
    title="Proportions finales – ALD (GMM difficile)"
)
