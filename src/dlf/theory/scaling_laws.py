"""
Scaling Laws Module for Neural Networks

This module implements the fundamental scaling law equations and visualization functions
for understanding how neural network performance scales with:
- Model size (number of parameters N)
- Dataset size (number of training tokens D)
- Compute budget (FLOPs C)

Based on:
- Kaplan et al. (2020): Scaling Laws for Neural Language Models
- Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)
- Hestness et al. (2017): Deep Learning Scaling is Predictable, Empirically
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


class ScalingLaws:
    """
    Implements the fundamental scaling law equations:
    L(N, D, C) ≈ L_∞ + A*N^(-α) + B*D^(-β) + E*C^(-γ)

    where:
    - L is the loss
    - N is the number of parameters
    - D is the number of training tokens
    - C is the compute budget (FLOPs)
    - L_∞ is the irreducible loss (noise floor)
    - A, B, E are amplitude constants
    - α, β, γ are scaling exponents
    """

    def __init__(self, L_inf=1.69, A=406.4, alpha=0.076,
                 B=410.7, beta=0.095, E=1.8, gamma=0.037):
        """
        Initialize scaling laws with default parameters from literature.

        Args:
            L_inf: Irreducible loss (noise floor)
            A, B, E: Amplitude constants for parameter, data, compute terms
            alpha, beta, gamma: Scaling exponents
        """
        self.L_inf = L_inf
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta
        self.E = E
        self.gamma = gamma

    def loss_from_params(self, N, noise=0.0):
        """
        Loss as function of parameter count (single-factor slice).

        Args:
            N: Number of parameters (can be array)
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Loss value(s)
        """
        loss = self.L_inf + self.A * N**(-self.alpha)
        if noise > 0:
            loss = loss + np.random.normal(0, noise, size=loss.shape if hasattr(loss, 'shape') else 1)
        return loss

    def loss_from_data(self, D, noise=0.0):
        """
        Loss as function of training tokens (single-factor slice).

        Args:
            D: Number of training tokens (can be array)
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Loss value(s)
        """
        loss = self.L_inf + self.B * D**(-self.beta)
        if noise > 0:
            loss = loss + np.random.normal(0, noise, size=loss.shape if hasattr(loss, 'shape') else 1)
        return loss

    def loss_from_compute(self, C, noise=0.0):
        """
        Loss as function of compute budget (single-factor slice).

        Args:
            C: Compute budget in FLOPs (can be array)
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Loss value(s)
        """
        loss = self.L_inf + self.E * C**(-self.gamma)
        if noise > 0:
            loss = loss + np.random.normal(0, noise, size=loss.shape if hasattr(loss, 'shape') else 1)
        return loss

    def loss_full(self, N, D, C=None, noise=0.0):
        """
        Full loss function combining all factors.

        Args:
            N: Number of parameters (can be array)
            D: Number of training tokens (can be array)
            C: Compute budget in FLOPs (optional, computed from N and D if None)
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Loss value(s)
        """
        if C is None:
            k = 6  # Approximate FLOPs per token per parameter (forward+backward)
            C = k * N * D

        loss = self.L_inf + self.A * N**(-self.alpha) + self.B * D**(-self.beta)
        if noise > 0:
            loss = loss + np.random.normal(0, noise, size=loss.shape if hasattr(loss, 'shape') else 1)
        return loss

    def compute_optimal_allocation(self, C):
        """
        Compute-optimal allocation following Chinchilla scaling laws.

        For a given compute budget C, this returns the optimal allocation
        of parameters N and training tokens D that minimizes loss.

        Args:
            C: Compute budget in FLOPs

        Returns:
            tuple: (N_optimal, D_optimal, eta)
                - N_optimal: Optimal number of parameters
                - D_optimal: Optimal number of training tokens
                - eta: Envelope exponent
        """
        # Envelope exponent: η = αβ/(α+β)
        eta = (self.alpha * self.beta) / (self.alpha + self.beta)

        # Allocation exponents (Chinchilla: both ≈ 0.5)
        exp_N = self.beta / (self.alpha + self.beta)
        exp_D = self.alpha / (self.alpha + self.beta)

        # Proportionality constants (simplified)
        k = 6  # FLOPs per token per parameter
        kappa_N = (self.B * self.beta / (self.A * self.alpha * k))**(1/(self.alpha + self.beta))
        kappa_D = (self.A * self.alpha * k / (self.B * self.beta))**(1/(self.alpha + self.beta))

        N_optimal = kappa_N * C**exp_N
        D_optimal = kappa_D * C**exp_D

        return N_optimal, D_optimal, eta

    def optimal_envelope(self, C, noise=0.0):
        """
        Loss on the compute-optimal frontier.

        This represents the best achievable loss for a given compute budget
        when parameters and data are optimally allocated.

        Args:
            C: Compute budget in FLOPs (can be array)
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Loss value(s) on the optimal frontier
        """
        N_opt, D_opt, eta = self.compute_optimal_allocation(C)

        # Simplified envelope: L*(C) = L_∞ + K*C^(-η)
        K = self.A * (self.alpha / (self.alpha + self.beta)) + \
            self.B * (self.beta / (self.alpha + self.beta))

        loss = self.L_inf + K * C**(-eta)
        if noise > 0:
            loss = loss + np.random.normal(0, noise, size=loss.shape if hasattr(loss, 'shape') else 1)
        return loss


def plot_single_factor_slices(scaling):
    """
    Reproduce the classic single-factor scaling plots.
    Shows how loss decreases with each resource on log-log axes.

    This creates three subplots showing:
    1. Loss vs Parameters (N)
    2. Loss vs Training Tokens (D)
    3. Loss vs Compute Budget (C)

    Args:
        scaling: ScalingLaws instance

    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    print("\n" + "="*70)
    print("PLOTTING SINGLE-FACTOR SCALING LAWS")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Parameter scaling
    N_range = np.logspace(6, 10, 50)  # 1M to 10B parameters
    loss_N = scaling.loss_from_params(N_range, noise=0.01)

    axes[0].loglog(N_range, loss_N, 'o-', markersize=4, linewidth=2, label='Observed')
    axes[0].loglog(N_range, scaling.loss_from_params(N_range), '--',
                   linewidth=2, alpha=0.7, label=f'Fit: L = {scaling.L_inf:.2f} + {scaling.A:.1f}N^(-{scaling.alpha:.3f})')
    axes[0].axhline(scaling.L_inf, color='red', linestyle=':', alpha=0.5, label=f'L∞ = {scaling.L_inf:.2f}')
    axes[0].set_xlabel('Parameters N', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss L(N)', fontsize=12, fontweight='bold')
    axes[0].set_title('Parameter Scaling Law', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Data scaling
    D_range = np.logspace(8, 12, 50)  # 100M to 1T tokens
    loss_D = scaling.loss_from_data(D_range, noise=0.01)

    axes[1].loglog(D_range, loss_D, 's-', markersize=4, linewidth=2, label='Observed')
    axes[1].loglog(D_range, scaling.loss_from_data(D_range), '--',
                   linewidth=2, alpha=0.7, label=f'Fit: L = {scaling.L_inf:.2f} + {scaling.B:.1f}D^(-{scaling.beta:.3f})')
    axes[1].axhline(scaling.L_inf, color='red', linestyle=':', alpha=0.5, label=f'L∞ = {scaling.L_inf:.2f}')
    axes[1].set_xlabel('Training Tokens D', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss L(D)', fontsize=12, fontweight='bold')
    axes[1].set_title('Data Scaling Law', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Compute scaling
    C_range = np.logspace(17, 23, 50)  # 10^17 to 10^23 FLOPs
    loss_C = scaling.loss_from_compute(C_range, noise=0.01)

    axes[2].loglog(C_range, loss_C, '^-', markersize=4, linewidth=2, label='Observed')
    axes[2].loglog(C_range, scaling.loss_from_compute(C_range), '--',
                   linewidth=2, alpha=0.7, label=f'Fit: L = {scaling.L_inf:.2f} + {scaling.E:.1f}C^(-{scaling.gamma:.3f})')
    axes[2].axhline(scaling.L_inf, color='red', linestyle=':', alpha=0.5, label=f'L∞ = {scaling.L_inf:.2f}')
    axes[2].set_xlabel('Compute C (FLOPs)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Loss L(C)', fontsize=12, fontweight='bold')
    axes[2].set_title('Compute Scaling Law', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print elasticities
    print("\n" + "="*70)
    print("ELASTICITIES (% change in loss per 10x increase in resource)")
    print("="*70)
    print(f"Parameter elasticity: ε_N = -α = -{scaling.alpha:.3f}")
    print(f"  → 10x more parameters reduces loss by {(1 - 10**(-scaling.alpha))*100:.1f}%")
    print(f"\nData elasticity: ε_D = -β = -{scaling.beta:.3f}")
    print(f"  → 10x more data reduces loss by {(1 - 10**(-scaling.beta))*100:.1f}%")
    print(f"\nCompute elasticity: ε_C = -γ = -{scaling.gamma:.3f}")
    print(f"  → 10x more compute reduces loss by {(1 - 10**(-scaling.gamma))*100:.1f}%")
    print("="*70)

    return fig, axes


def visualize_compute_optimal_frontier(scaling):
    """
    Show the compute-optimal allocation and compare different strategies.
    Reproduces the training curve envelope from Hoffmann et al. (2022).

    This creates three subplots showing:
    1. Optimal allocation of N and D vs compute budget
    2. Training curves for different model sizes
    3. Comparison of different allocation strategies

    Args:
        scaling: ScalingLaws instance

    Returns:
        fig: matplotlib figure object
    """
    print("\n" + "="*70)
    print("PLOTTING COMPUTE-OPTIMAL ALLOCATION (CHINCHILLA)")
    print("="*70)

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)

    # Compute budgets to explore
    C_budgets = np.logspace(18, 23, 30)

    # === Plot 1: Optimal Allocation ===
    ax1 = fig.add_subplot(gs[0, 0])

    N_optimal = []
    D_optimal = []
    for C in C_budgets:
        N_opt, D_opt, _ = scaling.compute_optimal_allocation(C)
        N_optimal.append(N_opt)
        D_optimal.append(D_opt)

    N_optimal = np.array(N_optimal)
    D_optimal = np.array(D_optimal)

    ax1.loglog(C_budgets, N_optimal, 'o-', linewidth=3, markersize=6, label='N*(C) ∝ C^0.50')
    ax1.loglog(C_budgets, D_optimal, 's-', linewidth=3, markersize=6, label='D*(C) ∝ C^0.50')
    ax1.set_xlabel('Compute Budget C (FLOPs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Allocation', fontsize=12, fontweight='bold')
    ax1.set_title('Chinchilla: Balanced Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Training Curves for Different Model Sizes ===
    ax2 = fig.add_subplot(gs[0, 1])

    model_sizes = [1e8, 5e8, 1e9, 5e9, 1e10]  # 100M to 10B
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_sizes)))

    for N, color in zip(model_sizes, colors):
        # For each model size, vary compute by varying training tokens
        C_range = np.logspace(18, 23, 50)
        D_range = C_range / (6 * N)  # Assuming k=6 FLOPs per token per param
        losses = scaling.loss_full(N, D_range, C_range, noise=0.005)

        ax2.loglog(C_range, losses, linewidth=2, color=color,
                   label=f'N={N/1e9:.1f}B')

    # Add optimal envelope
    envelope_losses = scaling.optimal_envelope(C_budgets, noise=0.005)
    ax2.loglog(C_budgets, envelope_losses, 'k--', linewidth=3,
               label='Optimal Envelope', alpha=0.8)

    ax2.set_xlabel('Compute C (FLOPs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Curve Envelope', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    # === Plot 3: Optimal vs Suboptimal Strategies ===
    ax3 = fig.add_subplot(gs[0, 2])

    # Strategy 1: Chinchilla (compute-optimal)
    loss_chinchilla = scaling.optimal_envelope(C_budgets)

    # Strategy 2: Over-parameterized (too big models, undertrained)
    # N ∝ C^0.73, D grows slowly
    N_overparam = 1e6 * (C_budgets / 1e18) ** 0.73
    D_overparam = C_budgets / (6 * N_overparam)
    loss_overparam = scaling.loss_full(N_overparam, D_overparam, C_budgets)

    # Strategy 3: Under-parameterized (small models, overtrained)
    # N ∝ C^0.3, D ∝ C^0.7
    N_underparam = 1e6 * (C_budgets / 1e18) ** 0.3
    D_underparam = 1e8 * (C_budgets / 1e18) ** 0.7
    loss_underparam = scaling.loss_full(N_underparam, D_underparam, C_budgets)

    # Strategy 4: Inference-optimal (for deployment)
    # Smaller models trained longer: N ∝ C^0.45, D ∝ C^0.55
    N_inference = 1e6 * (C_budgets / 1e18) ** 0.45
    D_inference = 1e8 * (C_budgets / 1e18) ** 0.55
    loss_inference = scaling.loss_full(N_inference, D_inference, C_budgets)

    ax3.loglog(C_budgets, loss_chinchilla, 'o-', linewidth=3, markersize=6,
               label='Compute-Optimal (Chinchilla)\nN,D ∝ C^0.50')
    ax3.loglog(C_budgets, loss_overparam, 's-', linewidth=2, markersize=5,
               label='Over-parameterized\nN ∝ C^0.73, D ∝ C^0.03')
    ax3.loglog(C_budgets, loss_underparam, '^-', linewidth=2, markersize=5,
               label='Under-parameterized\nN ∝ C^0.30, D ∝ C^0.70')
    ax3.loglog(C_budgets, loss_inference, 'd-', linewidth=2, markersize=5,
               label='Inference-Optimal\nN ∝ C^0.45, D ∝ C^0.55')

    ax3.set_xlabel('Compute Budget C (FLOPs)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Allocation Strategies Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print key insights
    print("\n" + "="*70)
    print("COMPUTE-OPTIMAL ALLOCATION INSIGHTS")
    print("="*70)
    _, _, eta = scaling.compute_optimal_allocation(1e20)
    print(f"Envelope exponent: η = αβ/(α+β) = {eta:.4f}")
    print(f"\nChinchilla Strategy (Compute-Optimal Training):")
    print(f"  • N ∝ C^{scaling.beta/(scaling.alpha+scaling.beta):.3f}")
    print(f"  • D ∝ C^{scaling.alpha/(scaling.alpha+scaling.beta):.3f}")
    print(f"  • Balanced scaling: double compute → ~1.4x both N and D")
    print(f"\nInference-Optimal Strategy (Deployment Focus):")
    print(f"  • N ∝ C^0.45 (smaller models)")
    print(f"  • D ∝ C^0.55 (trained longer)")
    print(f"  • Minimizes serving costs after training")
    print("="*70)

    return fig


def demonstrate_sample_efficiency(scaling):
    """
    Show that larger models are more sample-efficient:
    they achieve the same loss with fewer training tokens.

    This creates two subplots showing:
    1. Training curves for different model sizes
    2. Tokens needed vs model size to reach a target loss

    Args:
        scaling: ScalingLaws instance

    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    print("\n" + "="*70)
    print("DEMONSTRATING SAMPLE EFFICIENCY")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Model sizes to compare
    model_sizes = [1e8, 5e8, 1e9, 5e9, 1e10]  # 100M to 10B
    target_loss = 2.0  # Target loss to achieve

    # === Plot 1: Training curves showing data efficiency ===
    D_range = np.logspace(8, 12, 100)

    tokens_needed = []
    for N in model_sizes:
        losses = scaling.loss_full(N, D_range)
        axes[0].semilogx(D_range, losses, linewidth=2.5,
                        label=f'{N/1e9:.1f}B params')

        # Find tokens needed to reach target
        idx = np.argmin(np.abs(losses - target_loss))
        tokens_needed.append(D_range[idx])
        axes[0].scatter(D_range[idx], losses[idx], s=100, zorder=5)

    axes[0].axhline(target_loss, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Target Loss = {target_loss}')
    axes[0].set_xlabel('Training Tokens D', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Sample Efficiency: Larger Models Need Less Data',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # === Plot 2: Tokens needed vs model size ===
    axes[1].loglog(model_sizes, tokens_needed, 'o-', linewidth=3,
                   markersize=10, color='darkblue')

    # Fit power law
    log_N = np.log10(model_sizes)
    log_D = np.log10(tokens_needed)
    coeffs = np.polyfit(log_N, log_D, 1)
    slope = coeffs[0]

    axes[1].loglog(model_sizes, 10**(coeffs[1]) * np.array(model_sizes)**slope,
                   '--', linewidth=2, alpha=0.7, color='red',
                   label=f'Fit: D ∝ N^{slope:.2f}')

    axes[1].set_xlabel('Model Size N (parameters)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Tokens Needed for Loss={target_loss}',
                      fontsize=12, fontweight='bold')
    axes[1].set_title('Sample Efficiency Scaling', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Calculate efficiency gains
    baseline_tokens = tokens_needed[0]
    print("\n" + "="*70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("="*70)
    print(f"Target loss: {target_loss}")
    print(f"\nTokens needed to reach target:")
    for N, D in zip(model_sizes, tokens_needed):
        reduction = (1 - D/baseline_tokens) * 100
        speedup = baseline_tokens / D
        print(f"  {N/1e9:>6.1f}B params: {D/1e9:>8.1f}B tokens "
              f"({speedup:>4.1f}x faster, {reduction:>5.1f}% reduction)")

    print(f"\nKey Insight: A 10x larger model achieves the same loss with")
    print(f"            approximately {10**(-slope):.1f}x fewer training tokens!")
    print("="*70)

    return fig, axes


def print_summary():
    """Print comprehensive summary of scaling laws"""
    summary = """
╔══════════════════════════════════════════════════════════════════════╗
║                    SCALING LAWS: PRACTICAL GUIDELINES                ║
╚══════════════════════════════════════════════════════════════════════╝

📊 WHAT SCALING LAWS TELL US:

1. POWER LAW RELATIONSHIPS ARE PREDICTABLE
   • Loss decreases as L ∝ N^(-α), L ∝ D^(-β), L ∝ C^(-γ)
   • Exponents: α ≈ 0.076, β ≈ 0.095, γ ≈ 0.037
   • Log-log plots show straight lines → easy to extrapolate

2. CHINCHILLA SCALING (COMPUTE-OPTIMAL)
   • For training: N ∝ C^0.50, D ∝ C^0.50
   • Double compute → 1.4x larger model + 1.4x more data
   • Achieves best loss per FLOP during training

3. SAMPLE EFFICIENCY INCREASES WITH SIZE
   • Larger models need fewer tokens for same performance
   • 10x model size → ~5-10x fewer training examples
   • Frontloading capacity pays off in data efficiency

4. INFERENCE VS TRAINING OPTIMAL
   • Training-optimal: balanced N and D (Chinchilla)
   • Inference-optimal: smaller N, larger D (lower serving costs)
   • Choice depends on: training once vs serving millions

⚠️  WHAT SCALING LAWS DON'T TELL US:

1. Emergent capabilities and discontinuous improvements
2. Effects of data quality, diversity, and distribution shifts
3. Optimal architecture choices (depth, width, attention patterns)
4. Training stability, convergence, and optimization dynamics
5. Task-specific performance (only measure pre-training loss)

🎯 PRACTICAL RECOMMENDATIONS:

FOR RESEARCH (maximize final performance):
   → Use compute-optimal scaling (Chinchilla)
   → Iterate quickly with smaller pilots
   → Predict full-scale performance from small runs

FOR PRODUCTION (minimize serving costs):
   → Consider inference-optimal scaling
   → Trade training compute for lower per-query latency
   → Smaller models with more training

FOR CONSTRAINED BUDGETS:
   → Invest in data quality over raw quantity
   → Use learning rate schedules and regularization
   → Consider model distillation after training

╔══════════════════════════════════════════════════════════════════════╗
║  "Scaling laws are empirical regularities, not fundamental laws.     ║
║   They guide resource allocation but don't replace good judgment."   ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(summary)
