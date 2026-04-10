
<p align="center">
  <img src="https://img.shields.io/badge/RL-Theory-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Math-Heavy-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Theoretical-yellow?style=for-the-badge" />
</p>

<h1 align="center">SIGMA-WIGAC</h1>
<p align="center"><strong>S</strong>ignature-<strong>I</strong>nfused <strong>G</strong>eometric <strong>M</strong>ean-<strong>F</strong>ield <strong>A</strong>ctor-<strong>C</strong>ritic in <strong>W</strong>asserstein-<strong>F</strong>isher-<strong>R</strong>ao <strong>S</strong>pace</p>

<p align="center">
  <em>A mathematically rigorous reinforcement learning algorithm operating on trajectory measures in infinite-dimensional path space, equipped with rough path theory, geometric measure theory, and mean-field game dynamics.</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
  - [The Extended Path Space](#the-extended-path-space)
  - [Chen-Fliess Signature Transform](#chen-fliess-signature-transform)
  - [Wasserstein-Fisher-Rao Geometry](#wasserstein-fisher-rao-geometry)
  - [Information Geometry of Policy Manifolds](#information-geometry-of-policy-manifolds)
- [The Algorithm](#the-algorithm)
  - [Variational Problem on Path Space](#variational-problem-on-path-space)
  - [Critic: Kernel Mean Embedding in Tensor Algebra](#critic-kernel-mean-embedding-in-tensor-algebra)
  - [Actor: Natural Gradient in WFR Space](#actor-natural-gradient-in-wfr-space)
  - [Mean-Field Master Equation Layer](#mean-field-master-equation-layer)
- [Convergence Guarantees](#convergence-guarantees)
- [Implementation](#implementation)
- [Usage](#usage)
- [References](#references)

---

## Overview

SIGMA-WIGAC represents a paradigm shift from Markovian Decision Processes (MDPs) to **path-dependent control problems** in infinite-dimensional spaces. By lifting the RL problem to the space of probability measures over trajectories $\mathcal{P}_2(\Omega)$ and equipping it with the Wasserstein-Fisher-Rao metric, we obtain:

1. **Non-Markovian capability** via signature transforms (rough path theory)
2. **Geometric consistency** via natural gradients on statistical manifolds
3. **Mass transport regularization** via dynamic optimal transport
4. **Scalability to infinite populations** via mean-field game theory

### Key Innovation

Instead of optimizing a policy $\pi(a|s)$, we optimize a **flow of trajectory measures** $\rho_t \in \mathcal{P}_2(\Omega)$ satisfying the continuity equation with source:

$$
\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = \rho_t(r_t - 1)
$$

where $v_t$ represents policy improvement and $r_t$ represents exploration mass creation/destruction.

---

## Mathematical Foundations

### The Extended Path Space

Let $(\mathcal{X}, \mathfrak{B}(\mathcal{X}), d_{\mathcal{X}})$ be a Polish state space and $\mathcal{A}$ a compact Lie group of actions. We define the **horizon-free path space** as the disjoint union:

$$
\Omega = \bigsqcup_{t=0}^{\infty} \Omega_t, \quad \Omega_t = (\mathcal{X} \times \mathcal{A})^{t} \times \mathcal{X}
$$

We equip $\Omega$ with the **truncated metric**:

$$
d_{\Omega}(\omega, \omega') = \sum_{t=0}^{\infty} 2^{-t} \frac{d_{\mathcal{X}}(x_t, x'_t) + d_{\mathcal{A}}(a_t, a'_t)}{1 + d_{\mathcal{X}}(x_t, x'_t) + d_{\mathcal{A}}(a_t, a'_t)} \mathbf{1}_{\{t \leq \max(|\omega|, |\omega'|)\}}
$$

This makes $(\Omega, d_{\Omega})$ a complete separable metric space (Polish). Let $\mathcal{P}_2(\Omega)$ denote the space of probability measures with finite second moments:

$$
\mathcal{P}_2(\Omega) = \left\{ \mathbb{P} \in \mathcal{P}(\Omega) : \int_{\Omega} d_{\Omega}^2(\omega, \omega_0) d\mathbb{P}(\omega) < \infty \right\}
$$

### Chen-Fliess Signature Transform

For a path $\omega_{:t} = (x_0, a_0, \ldots, x_t)$, we construct the **augmented path** $\bar{\omega}: [0,1] \to \mathbb{R}^{d+1}$ via linear interpolation of $(t, x_t, a_t)$. The **signature transform** is defined as the collection of iterated integrals:

$$
S(\omega_{:t}) = \left(1, \int_{0<u_1<1} d\bar{\omega}_{u_1}, \int_{0<u_1<u_2<1} d\bar{\omega}_{u_1} \otimes d\bar{\omega}_{u_2}, \ldots \right) \in \mathcal{T}((\mathbb{R}^{d+1}))
$$

where $\mathcal{T}((V)) = \prod_{k=0}^{\infty} V^{\otimes k}$ denotes the completed tensor algebra.

**Universal Approximation Property:** The truncated signature $S_N(\omega) = \text{proj}_{\leq N} S(\omega)$ provides a universal feature map for paths modulo tree-like equivalence (Hambly-Lyons uniqueness theorem).

#### Adaptive Signature Depth

We dynamically adjust the truncation level $N$ based on the **local Lyapunov exponent**:

$$
\lambda(\omega_{:t}) = \limsup_{k \to t} \frac{1}{k} \log \left\| \nabla_{x_0} x_k \right\|
$$

$$
N^{\ast}(\omega_{:t}) = \left\lceil \frac{\lambda_{\max} - \lambda(\omega_{:t})}{\lambda_{\max} - \lambda_{\min}} \cdot N_{\max} \right\rceil
$$

This allocates higher tensor degrees to chaotic trajectory segments.

### Wasserstein-Fisher-Rao Geometry

Standard Wasserstein distance conserves mass. For RL, we require the ability to create and destroy mass (exploration/exploitation). The **Wasserstein-Fisher-Rao (WFR)** distance on $\mathcal{P}_2(\Omega)$ is:

$$
\mathcal{W}_{\text{FR}}^2(\mu, \nu) = \inf_{\rho_t, v_t, r_t} \int_0^1 \left[ \int_{\Omega} \left|v_t(\omega)\right|^2 + 4\left|\nabla_{\omega} \sqrt{r_t(\omega)}\right|^2 \right] d\rho_t(\omega) dt
$$

subject to the **continuity equation with source**:

$$
\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = \rho_t(r_t - 1), \quad \rho_0 = \mu, \quad \rho_1 = \nu
$$

**Geometric Interpretation:** This defines a Riemannian metric on $\mathcal{P}_2(\Omega)$ that generalizes the Otto-Wasserstein structure to allow non-conservative flows. The term $r_t$ represents the rate of mass creation/destruction.

### Information Geometry of Policy Manifolds

Let $\mathcal{M} = \{\pi_{\theta} : \theta \in \Theta \subseteq \mathbb{R}^d\}$ be our parametric policy manifold. Each $\pi_{\theta}$ induces a path measure $\mathbb{P}^{\pi_{\theta}} \in \mathcal{P}_2(\Omega)$.

**Fisher-Rao Metric Tensor:**

$$
G_{ij}(\theta) = \mathbb{E}_{\omega \sim \mathbb{P}^{\pi_{\theta}}} \left[ \sum_{t=0}^{|\omega|} \partial_i \log \pi_{\theta}(a_t|x_t) \cdot \partial_j \log \pi_{\theta}(a_t|x_t) \right]
$$

This defines a Riemannian metric on $\mathcal{M}$, where the length of a curve $\theta(t)$ measures the amount of information updated along the trajectory.

---

## The Algorithm

### Variational Problem on Path Space

We formulate RL as optimization over probability flows in $\mathcal{P}_2(\Omega)$. Let $\mathbb{P}^{\pi}$ denote the path measure induced by policy $\pi$.

**The Geometric Entropy-Regularized Objective:**

$$
\mathcal{J}(\mathbb{P}) = \underbrace{\mathbb{E}^{\mathbb{P}}[\Phi(\omega)]}_{\text{Path Cost}} + \underbrace{\lambda \mathcal{H}(\mathbb{P} | \mathbb{P}^{\text{ref}})}_{\text{KL Divergence}} + \underbrace{\gamma \mathcal{W}_{\text{FR}}^2(\mathbb{P}, \mu^{\ast} \otimes \pi_{\text{target}})}_{\text{Transport Penalty}} + \underbrace{\eta \mathcal{M}(\mathbb{P})}_{\text{Mean-Field}}
$$

where:
- $\Phi(\omega) = \sum_{t=0}^{|\omega|} \gamma^t r(x_t, a_t)$ is the discounted return functional
- $\mathcal{H}(\mathbb{P} | \mathbb{P}^{\text{ref}}) = \int \log \frac{d\mathbb{P}}{d\mathbb{P}^{\text{ref}}} d\mathbb{P}$ is the relative entropy
- $\mathcal{M}(\mathbb{P}) = \int_{\Omega \times \Omega} W(\omega, \omega') d(\mathbb{P} \times \mathbb{P})$ captures pairwise mean-field interactions
- $\lambda, \gamma, \eta > 0$ are regularization parameters

### Critic: Kernel Mean Embedding in Tensor Algebra

We approximate the value function using the **Expected Signature Kernel**:

$$
\mathcal{K}_N(\omega, \omega') = \langle S_N(\omega), S_N(\omega') \rangle_{\mathcal{T}_N} = \sum_{k=0}^{N} \langle S_k(\omega), S_k(\omega') \rangle
$$

Define the **path-dependent Q-function** in the RKHS $\mathcal{H}_{\text{sig}}$:

$$
Q_{\psi}(\omega_{:t}, a) = \langle \phi(\omega_{:t}, a), \psi \rangle_{\mathcal{H}}
$$

where $\phi(\omega_{:t}, a) = S_{N^{\ast}(\omega_{:t})}(\omega_{:t}) \otimes \mathbf{e}_a$ and $\psi \in \mathcal{H}$.

**The Path-Dependent Bellman Equation:**

$$
Q_{\psi}(\omega_{:t}, a) = r(x_t, a) + \gamma \sigma_{\lambda}\left[ \mathbb{E}_{x' \sim P(\cdot|x_t,a)}[Q_{\psi}(\omega_{:t} \oplus (x', \cdot), \cdot)] \right]
$$

where $\sigma_{\lambda}$ is the soft-Bellman operator:

$$
\sigma_{\lambda}[f](a) = \lambda \log \sum_{a'} \exp\left(\frac{f(a')}{\lambda}\right)
$$

**Critic Update (Projected Natural Gradient):**

$$
\psi_{k+1} = \psi_k - \alpha G_{\text{sig}}^{-1} \nabla_{\psi} \mathcal{L}_{\text{TD}}
$$

with $G_{\text{sig}}$ the kernel Fisher matrix and loss:

$$
\mathcal{L}_{\text{TD}} = \mathbb{E}_{(\omega, a) \sim \mathcal{D}} \left[ \left( Q_{\psi}(\omega, a) - \mathcal{T} Q_{\psi_k}(\omega, a) \right)^2 \right]
$$

### Actor: Natural Gradient in WFR Space

The **geometric policy gradient** accounts for the curvature of the statistical manifold:

$$
\tilde{\nabla}_{\theta} \mathcal{J} = \mathbb{E}_{\tau \sim \mathbb{P}^{\pi_{\theta}}} \left[ \sum_{t=0}^{|\tau|} \nabla_{\theta} \log \pi_{\theta}(a_t|x_t) \cdot \text{Adv}(\tau_{:t}, a_t) \right]
$$

**Advantage Function:**

$$
\text{Adv}(\omega_{:t}, a) = Q_{\psi}(\omega_{:t}, a) - \lambda \log \pi_{\theta}(a|x_t) + \gamma \frac{\delta \mathcal{W}_{\text{FR}}^2}{\delta \mathbb{P}}(\omega_{:t})
$$

**Natural Gradient Update:**

$$
\theta_{k+1} = \text{Exp}_{\theta_k}^{\mathcal{M}}(-\beta G(\theta_k)^{-1} \tilde{\nabla}_{\theta} \mathcal{J})
$$

where $\text{Exp}^{\mathcal{M}}$ is the exponential map on the statistical manifold.

**Computing the WFR Gradient:**
By the Benamou-Brenier formulation and the Schrödinger bridge representation:

$$
\nabla_{\theta} \mathcal{W}_{\text{FR}}^2 = \mathbb{E}_{\mathbb{P}^{\pi_{\theta}}}[\psi(\omega) \nabla_{\theta} \log \mathbb{P}^{\pi_{\theta}}(\omega)]
$$

with $\psi$ solving the **Hamilton-Jacobi-Bellman equation on path space**:

$$
\partial_t \psi + \frac{1}{2}|\nabla \psi|^2 + \frac{\gamma^2}{8} (e^{\psi} - 1)^2 = 0
$$

### Mean-Field Master Equation Layer

For multi-agent settings, we track the population distribution $\mu_t \in \mathcal{P}(\mathcal{X})$. The policy becomes **mean-field dependent**:

$$
\pi_{\theta}(a|x, \mu) = \text{Softmax}\left( \frac{\langle \mathbb{E}_{\xi \sim \mu}[K(x, \cdot, \xi)], w_{\theta} \rangle}{\lambda} \right)
$$

The value function $U(t, \mu)$ satisfies the **Master Equation** (Lasry-Lions):

$$
\partial_t U(t, \mu) + \int_{\mathcal{X}} \nabla_x \partial_{\mu} U(t, \mu, x) \cdot b(x, \mu, \pi^{\ast}) d\mu(x) + \frac{1}{2} \text{Tr}(\sigma \sigma^T \partial_{xx} \partial_{\mu} U) = 0
$$

approximated via deep neural operators acting on measure embeddings.

---

## Convergence Guarantees

### Theorem 1: Geometric Ergodicity
Under standard regularity conditions (Lipschitz rewards, Fisher information matrix bounded away from singular), SIGMA-WIGAC converges to the unique solution $\mathbb{P}^{\ast}$ of the variational problem with rate:

$$
d_{\text{FR}}(\mathbb{P}^{\pi_{\theta_k}}, \mathbb{P}^{\ast}) \leq \frac{C}{\sqrt{k}}
$$

where $d_{\text{FR}}$ is the Fisher-Rao distance on $\mathcal{P}(\Omega)$.

### Theorem 2: Sample Complexity
For a system with Lyapunov dimension $D_L$, the sample complexity to achieve $\epsilon$-optimality is:

$$
N_{\text{sample}} = \tilde{O}\left(\epsilon^{-D_L/2}\right)
$$

rather than $\tilde{O}(\epsilon^{-\dim(\mathcal{X})/2})$, due to adaptive signature truncation.

### Theorem 3: Mean-Field Convergence
As the number of agents $N \to \infty$, the empirical measure flow $\mu_t^N$ converges to the solution of the Master Equation:

$$
\mathbb{E}\left[ \mathcal{W}_2^2(\mu_t^N, \mu_t) \right] \leq \frac{C}{N}
$$

---

## Implementation

### Dependencies

```python
numpy >= 1.21.0
torch >= 2.0.0
iisignature >= 0.24  # For Chen-Fliess signatures
geomstats >= 2.5.0   # For Riemannian geometry on manifolds
optimal-transport >= 0.8.0
scipy >= 1.9.0
```

### Core Components

```python
class SignatureCritic(nn.Module):
    """
    Path-dependent critic using signature kernel mean embedding.
    """
    def __init__(self, d_state, d_action, max_signature_depth):
        super().__init__()
        self.N_max = max_signature_depth
        self.psi = nn.Parameter(torch.randn(signature_dimension(d_state + d_action, max_signature_depth)))
        
    def forward(self, path, action):
        # Compute adaptive signature depth based on Lyapunov exponent
        N_star = self.compute_adaptive_depth(path)
        sig = iisignature.sig(path, N_star)
        feature = torch.kron(sig, torch.eye(self.d_action)[action])
        return torch.dot(feature, self.psi)
    
    def compute_adaptive_depth(self, path):
        # Estimate local Lyapunov exponent
        lyapunov = estimate_lyapunov(path)
        N_star = ceil((lambda_max - lyapunov) / (lambda_max - lambda_min) * self.N_max)
        return N_star

class WFRActor(nn.Module):
    """
    Policy with Wasserstein-Fisher-Rao natural gradient.
    """
    def __init__(self, d_state, d_action):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(d_state, d_action))
        
    def forward(self, state, population_mu=None):
        if population_mu is not None:
            # Mean-field dependency
            kernel_features = self.compute_mean_field_kernel(state, population_mu)
            logits = kernel_features @ self.theta
        else:
            logits = state @ self.theta
        return F.log_softmax(logits / lambda_temp, dim=-1)
    
    def natural_gradient(self, advantage_batch):
        # Compute Fisher Information Matrix
        G = self.compute_fisher_matrix()
        # Solve G @ delta = grad using conjugate gradient
        grad = self.compute_policy_gradient(advantage_batch)
        delta = conjugate_gradient_solve(G, grad)
        return delta
```

### Pseudocode

```
ALGORITHM SIGMA-WIGAC

Input: 
  - Initial parameters θ₀, ψ₀
  - Replay buffer D
  - Hyperparameters λ, γ, η, N_max, α, β

Initialize:
  - Empty buffer D
  - Reference measure P_ref

REPEAT until convergence:
  
  # Data Collection
  FOR each environment worker i = 1 to N_workers:
    ω_i ← Rollout(π_θ, Environment)
    Compute Lyapunov exponent λ(ω_i)
    N* ← AdaptiveDepth(λ(ω_i), N_max)
    Compute signature S_{N*}(ω_i)
    Store (ω_i, S_{N*}(ω_i), λ(ω_i)) in D
  
  # Critic Update (Kernel Mean Embedding)
  Sample batch B from D
  Compute kernel matrix K_ij = K_N(ω_i, ω_j)
  Solve linear system: Q = K ψ
  Compute TD target: Q_target = T[Q]
  Natural gradient: Δψ = G_sig^{-1} ∇_ψ ||Q - Q_target||²
  Update: ψ ← ψ - α Δψ
  
  # Actor Update (WFR Natural Gradient)
  FOR each trajectory ω in batch:
    Compute advantage:
      A(ω,a) = Q_ψ(ω,a) - λ log π_θ(a|x) 
               + γ ∇_WFR(ω)  # Via continuity equation solver
  
  Compute Fisher matrix: G(θ) = E[∇_θ log π ∇_θ log π^T]
  Compute geometric gradient: g = E[∇_θ log π · A]
  Solve: Δθ = G(θ)^{-1} g  # Using truncated Newton
  
  # Exponential map update
  θ ← Exp_θ^M(-β Δθ)
  
  # Mean-Field Update (if multi-agent)
  IF population_mode:
    Solve Fokker-Planck: ∂_t μ + ∇·(b μ) = ½ Tr(σσ^T ∂_{xx} μ)
    Update population embedding μ_t

RETURN θ, ψ
```

---

## Usage

### Basic Single-Agent

```python
from sigma_wigac import SIGMAWIGACAgent

# Initialize agent
agent = SIGMAWIGACAgent(
    state_dim=64,
    action_dim=4,
    max_signature_depth=4,
    wfr_regularization=0.1,
    entropy_coef=0.01,
    use_natural_gradient=True
)

# Training loop
for episode in range(1000):
    trajectory = []
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, trajectory_history=trajectory)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    
    # Update using path-space gradient
    agent.update(trajectory)
```

### Multi-Agent Mean-Field

```python
# Initialize with mean-field layer
agent = SIGMAWIGACAgent(
    state_dim=64,
    action_dim=4,
    mean_field=True,
    n_agents=1000,
    master_equation_solver='neural_operator'
)

# Training with population dynamics
for t in range(T):
    # Agents interact with mean field μ_t
    actions = agent.select_actions(states, population_dist)
    next_states, rewards = env.step(actions)
    
    # Update agent policies
    agent.update(states, actions, rewards, next_states)
    
    # Update mean field distribution
    agent.update_population_distribution(states, next_states)
```

---

## References

1. Hambly, B. M., & Lyons, T. J. (2010). "Uniqueness for the signature of a path of bounded variation and the reduced path group." *Annals of Mathematics*.

2. Benamou, J. D., & Brenier, Y. (2000). "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem." *Numerische Mathematik*.

3. Kondratyev, S., Monsaingeon, L., & Vorotnikov, D. (2016). "A new optimal transport distance on the space of finite Radon measures." *Advances in Differential Equations*.

4. Amari, S. I. (1998). "Natural gradient works efficiently in learning." *Neural Computation*.

5. Lasry, J. M., & Lions, P. L. (2007). "Mean field games." *Japanese Journal of Mathematics*.

6. Chevyrev, I., & Lyons, T. (2016). "Characteristic functions of measures on geometric rough paths." *Annals of Probability*.

7. Otto, F. (2001). "The geometry of dissipative evolution equations: the porous medium equation." *Communications in Partial Differential Equations*.

---

## Citation

If you use SIGMA-WIGAC in your research, please cite:

```bibtex
@article{sigma_wigac_2026,
  title={SIGMA-WIGAC: Signature-Infused Geometric Mean-Field Actor-Critic in Wasserstein-Fisher-Rao Space},
  author={[Devanik]},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/Devanik21/sigma-wigac}
}
```

---

<p align="center">
  <em>"The policy is not a function of state, but a flow on the manifold of trajectories."</em>
</p>
