"""
Utility functions for Part 3: Spectral Analysis of Fine-tuned Language Models

This module contains helper functions for:
- Data loading and preprocessing
- Model training and evaluation
- WeightWatcher spectral analysis
- Visualization

Author: Homework 1 Part 3
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import torch
import requests
import random
import os
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM
import weightwatcher as ww


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def split_train_test(
    data: List,
    test_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[List, List]:
    """
    Shuffle and split data into (train, test) only.

    Args:
        data: List of data items to split
        test_fraction: Fraction of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_list, test_list)

    Example:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> train, test = split_train_test(data, test_fraction=0.2, seed=42)
        >>> len(test) / len(data)
        0.2
    """
    assert 0 < test_fraction < 1, "test_fraction must be in (0, 1)"

    items = list(data)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_test = int(n * test_fraction)
    n_train = n - n_test

    train = items[:n_train]
    test = items[n_train:]

    return train, test


def download_ir_triplets(
    url: str = None,
    test_fraction: float = 0.2,
    seed: int = 42
) -> Dict[str, List]:
    """
    Download and split the IR Triplets dataset into train/test.

    The IR Triplets dataset contains inductive reasoning examples with:
    - Training Observations (Context)
    - Question
    - Answer

    Args:
        url: URL to download dataset from (default: IR Triplets GitHub)
        test_fraction: Fraction of data to use for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with 'train' and 'test' keys containing lists of examples

    Example:
        >>> triplets = download_ir_triplets()
        >>> print(f"Train: {len(triplets['train'])}, Test: {len(triplets['test'])}")
        Train: 1446, Test: 361
    """
    if url is None:
        url = "https://raw.githubusercontent.com/omroot/InductiveSLM/master/cache/raw_data/ir_triplets/ir_triplets.json"

    print(f"Downloading from: {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    train, test = split_train_test(data=data, test_fraction=test_fraction, seed=seed)

    return {'train': train, 'test': test}


def format_example(ex: Dict[str, str]) -> Dict[str, str]:
    """
    Format example for causal LM training.

    Combines Context (Training Observations), Question, and Answer into
    a single text string for next-token prediction.

    Format: "Context: {C}\n\nQuestion: {Q}\n\nAnswer: {A}"

    Args:
        ex: Dictionary with keys 'Training Observations', 'Question', 'Answer'

    Returns:
        Dictionary with key 'text' containing formatted string

    Example:
        >>> ex = {
        ...     "Training Observations": "The series is trending upward.",
        ...     "Question": "What happens next?",
        ...     "Answer": "Continued growth."
        ... }
        >>> formatted = format_example(ex)
        >>> print(formatted['text'][:50])
        Context: The series is trending upward.

Question
    """
    obs = ex["Training Observations"].strip()
    q = ex["Question"].strip()
    a = ex["Answer"].strip()

    # Complete text for causal LM
    text = f"Context: {obs}\n\nQuestion: {q}\n\nAnswer: {a}"

    return {"text": text}


def format_dataset(data: List[Dict]) -> List[Dict]:
    """
    Format an entire dataset for causal LM training.

    Args:
        data: List of examples to format

    Returns:
        List of formatted examples with 'text' key

    Example:
        >>> data = [{"Training Observations": "...", "Question": "...", "Answer": "..."}]
        >>> formatted = format_dataset(data)
        >>> 'text' in formatted[0]
        True
    """
    return [format_example(ex) for ex in data]


# ============================================================================
# WeightWatcher Analysis
# ============================================================================

def analyze_model_with_weightwatcher(
    model,
    model_name: str = "model",
    save_details: bool = True,
    device: torch.device = None
) -> Dict:
    """
    Analyze a PyTorch model using WeightWatcher.

    WeightWatcher performs spectral analysis on weight matrices to compute
    the power law exponent (alpha) for each layer. This metric indicates:
    - 2 ≤ α ≤ 6: Well-trained layer
    - α < 2: Over-trained (memorization)
    - α > 6: Under-trained (needs more training)

    Args:
        model: PyTorch model to analyze
        model_name: Name for saving results (default: "model")
        save_details: Whether to save detailed CSV (default: True)
        device: Device to return model to after analysis (default: None = CPU)

    Returns:
        Dictionary containing:
        - 'model_name': Name of the model
        - 'summary': WeightWatcher summary
        - 'details': Full DataFrame of layer-wise analysis
        - 'alpha_mean': Mean alpha across all layers
        - 'alpha_median': Median alpha
        - 'alpha_std': Standard deviation of alpha
        - 'n_layers': Number of layers analyzed
        - 'n_good_layers': Number of well-trained layers (2 ≤ α ≤ 6)
        - 'n_overtrained': Number of over-trained layers (α < 2)
        - 'n_undertrained': Number of under-trained layers (α > 6)

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> results = analyze_model_with_weightwatcher(model, "distilgpt2")
        >>> print(f"Mean alpha: {results['alpha_mean']:.3f}")
        Mean alpha: 4.028
    """
    if device is None:
        device = torch.device('cpu')

    print(f"\nAnalyzing {model_name} with WeightWatcher...")

    # Move model to CPU for WeightWatcher (it requires CPU)
    model_cpu = model.cpu()

    # Initialize WeightWatcher
    watcher = ww.WeightWatcher(model=model_cpu)

    # Analyze
    details = watcher.analyze(plot=False)
    summary = watcher.get_summary()

    # Extract key metrics
    results = {
        'model_name': model_name,
        'summary': summary,
        'details': details,
        'layer_names': details['layer_name'].tolist() if 'layer_name' in details.columns else [],
        'alpha_values': details['alpha'].tolist() if 'alpha' in details.columns else [],
    }

    # Calculate statistics
    alpha_values = results['alpha_values']
    if alpha_values:
        results['alpha_mean'] = np.mean(alpha_values)
        results['alpha_median'] = np.median(alpha_values)
        results['alpha_std'] = np.std(alpha_values)
        results['n_layers'] = len(alpha_values)
        results['n_good_layers'] = sum(1 for a in alpha_values if 2 <= a <= 6)
        results['n_overtrained'] = sum(1 for a in alpha_values if a < 2)
        results['n_undertrained'] = sum(1 for a in alpha_values if a > 6)

    # Save detailed results
    if save_details:
        details.to_csv(f'weightwatcher_{model_name}.csv', index=False)
        print(f"  ✓ Saved to weightwatcher_{model_name}.csv")

    # Print summary
    print(f"  ✓ Analysis complete")
    print(f"    Layers analyzed: {results.get('n_layers', 0)}")
    if 'alpha_mean' in results:
        print(f"    Mean α: {results['alpha_mean']:.3f}")
        print(f"    Good layers (2 ≤ α ≤ 6): {results['n_good_layers']}/{results['n_layers']}")

    # Move model back to device
    model.to(device)

    return results


def analyze_all_checkpoints(
    output_dir: str,
    trainer,
    test_dataset,
    data_collator,
    training_args,
    device: torch.device = None,
    analyze_every_n: int = 2
) -> Dict[str, Dict]:
    """
    Analyze all saved model checkpoints with WeightWatcher.

    Args:
        output_dir: Directory containing checkpoint folders
        trainer: HuggingFace Trainer instance
        test_dataset: Test dataset for evaluation
        data_collator: Data collator for batching
        training_args: Training arguments
        device: Device to use (default: CPU)
        analyze_every_n: Analyze every Nth checkpoint (default: 2)

    Returns:
        Dictionary mapping checkpoint names to analysis results

    Example:
        >>> results = analyze_all_checkpoints(
        ...     "./finetuned_model",
        ...     trainer,
        ...     test_dataset,
        ...     data_collator,
        ...     training_args
        ... )
        >>> len(results)
        3
    """
    from transformers import Trainer

    if device is None:
        device = torch.device('cpu')

    print("="*80)
    print("ANALYZING ALL CHECKPOINTS")
    print("="*80)

    # Find all checkpoint directories
    checkpoint_paths = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_paths.append(os.path.join(output_dir, item))

    checkpoint_paths.sort(key=lambda x: int(x.split('-')[-1]))

    print(f"Found {len(checkpoint_paths)} checkpoints")
    print(f"Analyzing every {analyze_every_n} checkpoint(s) to save time...")
    print("-"*80)

    checkpoint_results = {}

    # Analyze every Nth checkpoint to save time
    for i, checkpoint_path in enumerate(checkpoint_paths[::analyze_every_n]):
        checkpoint_num = checkpoint_path.split('-')[-1]
        print(f"\nCheckpoint {i+1}: {checkpoint_path}")

        # Load checkpoint
        checkpoint_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

        # Evaluate checkpoint
        trainer_temp = Trainer(
            model=checkpoint_model,
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        eval_results = trainer_temp.evaluate()

        print(f"  Loss: {eval_results.get('eval_loss', 0):.4f}")
        print(f"  Perplexity: {np.exp(eval_results.get('eval_loss', 0)):.2f}")

        # Analyze with WeightWatcher
        checkpoint_name = f"checkpoint_{checkpoint_num}"
        results = analyze_model_with_weightwatcher(
            checkpoint_model,
            model_name=checkpoint_name,
            device=device
        )
        results['loss'] = eval_results.get('eval_loss', 0)
        results['perplexity'] = np.exp(eval_results.get('eval_loss', 0))
        results['checkpoint_number'] = int(checkpoint_num)

        checkpoint_results[checkpoint_name] = results

        # Clean up
        del checkpoint_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return checkpoint_results


# ============================================================================
# Visualization
# ============================================================================

def prepare_visualization_data(checkpoint_results: Dict[str, Dict]) -> Dict[str, List]:
    """
    Prepare data for visualization from checkpoint results.

    Args:
        checkpoint_results: Dictionary mapping checkpoint names to results

    Returns:
        Dictionary containing lists of:
        - checkpoint_names
        - checkpoint_numbers
        - alpha_means
        - losses
        - perplexities
        - n_good_layers_list

    Example:
        >>> results = {...}  # checkpoint results
        >>> viz_data = prepare_visualization_data(results)
        >>> len(viz_data['checkpoint_names'])
        4
    """
    checkpoint_names = []
    checkpoint_numbers = []
    alpha_means = []
    losses = []
    perplexities = []
    n_good_layers_list = []

    for name, results in sorted(checkpoint_results.items(),
                               key=lambda x: x[1].get('checkpoint_number', -1)):
        if 'alpha_mean' in results:
            checkpoint_names.append(name)
            checkpoint_numbers.append(results.get('checkpoint_number', 0))
            alpha_means.append(results['alpha_mean'])
            losses.append(results.get('loss', 0))
            perplexities.append(results.get('perplexity', 0))
            n_good_layers_list.append(results['n_good_layers'])

    return {
        'checkpoint_names': checkpoint_names,
        'checkpoint_numbers': checkpoint_numbers,
        'alpha_means': alpha_means,
        'losses': losses,
        'perplexities': perplexities,
        'n_good_layers_list': n_good_layers_list
    }


def create_spectral_analysis_plots(
    viz_data: Dict[str, List],
    save_path: str = 'spectral_analysis.png',
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150
):
    """
    Create 4-panel visualization of spectral analysis results.

    Plots:
    1. Power law exponent (alpha) evolution over training
    2. Test loss evolution over training
    3. Perplexity evolution over training
    4. Correlation between alpha and loss

    Args:
        viz_data: Dictionary from prepare_visualization_data()
        save_path: Path to save figure (default: 'spectral_analysis.png')
        figsize: Figure size in inches (default: (16, 12))
        dpi: Resolution in dots per inch (default: 150)

    Returns:
        matplotlib Figure object

    Example:
        >>> viz_data = prepare_visualization_data(checkpoint_results)
        >>> fig = create_spectral_analysis_plots(viz_data, 'results.png')
    """
    import matplotlib.pyplot as plt

    checkpoint_numbers = viz_data['checkpoint_numbers']
    alpha_means = viz_data['alpha_means']
    losses = viz_data['losses']
    perplexities = viz_data['perplexities']

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Mean alpha vs checkpoint
    ax1 = axes[0, 0]
    ax1.plot(checkpoint_numbers, alpha_means, 'o-', linewidth=2.5,
             markersize=10, color='#3498db', label='Mean α')
    ax1.axhline(y=2, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=6, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.fill_between(checkpoint_numbers, 2, 6, alpha=0.2, color='green')
    ax1.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean α', fontsize=12, fontweight='bold')
    ax1.set_title('Power Law Exponent Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs checkpoint
    ax2 = axes[0, 1]
    ax2.plot(checkpoint_numbers, losses, 's-', linewidth=2.5,
             markersize=10, color='#e74c3c', label='Test Loss')
    ax2.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Perplexity vs checkpoint
    ax3 = axes[1, 0]
    ax3.plot(checkpoint_numbers, perplexities, '^-', linewidth=2.5,
             markersize=10, color='#9b59b6', label='Perplexity')
    ax3.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax3.set_title('Perplexity Evolution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Alpha vs Loss correlation
    ax4 = axes[1, 1]
    ax4.scatter(alpha_means, losses, s=200, alpha=0.6,
               c=checkpoint_numbers, cmap='viridis',
               edgecolors='black', linewidth=2)
    for i, (alpha, loss) in enumerate(zip(alpha_means, losses)):
        ax4.annotate(f'{checkpoint_numbers[i]}', (alpha, loss),
                    fontsize=9, ha='center', va='bottom')

    if len(alpha_means) > 2:
        z = np.polyfit(alpha_means, losses, 1)
        p = np.poly1d(z)
        alpha_range = np.linspace(min(alpha_means), max(alpha_means), 100)
        ax4.plot(alpha_range, p(alpha_range), "r--", alpha=0.8, linewidth=2,
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

    ax4.set_xlabel('Mean α', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax4.set_title('Correlation: α vs Loss', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved {save_path}")

    return fig


def create_summary_table(
    checkpoint_results: Dict[str, Dict],
    save_path: str = 'weightwatcher_summary.csv'
) -> pd.DataFrame:
    """
    Create summary table of checkpoint analysis results.

    Args:
        checkpoint_results: Dictionary mapping checkpoint names to results
        save_path: Path to save CSV (default: 'weightwatcher_summary.csv')

    Returns:
        pandas DataFrame with summary statistics

    Example:
        >>> summary = create_summary_table(checkpoint_results)
        >>> print(summary.columns)
        Index(['Checkpoint', 'CP#', 'Loss', 'Perplexity', 'Mean α', 'Good Layers'], dtype='object')
    """
    viz_data = prepare_visualization_data(checkpoint_results)
    checkpoint_names = viz_data['checkpoint_names']

    summary_data = []
    for name in checkpoint_names:
        results = checkpoint_results[name]
        summary_data.append({
            'Checkpoint': name,
            'CP#': results.get('checkpoint_number', 0),
            'Loss': f"{results.get('loss', 0):.4f}",
            'Perplexity': f"{results.get('perplexity', 0):.2f}",
            'Mean α': f"{results['alpha_mean']:.3f}",
            'Good Layers': f"{results['n_good_layers']}/{results['n_layers']}",
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    if save_path:
        summary_df.to_csv(save_path, index=False)
        print(f"\n✓ Summary saved to {save_path}")

    return summary_df


def print_key_findings(viz_data: Dict[str, List]):
    """
    Print key findings from the analysis.

    Args:
        viz_data: Dictionary from prepare_visualization_data()

    Example:
        >>> viz_data = prepare_visualization_data(checkpoint_results)
        >>> print_key_findings(viz_data)
    """
    checkpoint_names = viz_data['checkpoint_names']
    alpha_means = viz_data['alpha_means']
    losses = viz_data['losses']
    perplexities = viz_data['perplexities']

    print("="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Best checkpoint by loss
    best_loss_idx = np.argmin(losses)
    print(f"\n1. BEST PERFORMING CHECKPOINT (Lowest Loss):")
    print(f"   Checkpoint: {checkpoint_names[best_loss_idx]}")
    print(f"   Loss: {losses[best_loss_idx]:.4f}")
    print(f"   Perplexity: {perplexities[best_loss_idx]:.2f}")
    print(f"   Mean α: {alpha_means[best_loss_idx]:.3f}")

    # Correlation
    if len(alpha_means) > 2:
        corr = np.corrcoef(alpha_means, losses)[0, 1]
        print(f"\n2. CORRELATION ANALYSIS:")
        print(f"   α vs Loss: {corr:.3f}")

        if corr < -0.5:
            print(f"   Strong negative correlation: Lower α → Higher loss")
        elif corr > 0.5:
            print(f"   Strong positive correlation: Higher α → Higher loss")
        else:
            print(f"   Weak/moderate correlation")

    # Evolution
    print(f"\n3. TRAINING EVOLUTION:")
    print(f"   Initial mean α: {alpha_means[0]:.3f}")
    print(f"   Final mean α: {alpha_means[-1]:.3f}")
    print(f"   Change: {alpha_means[-1] - alpha_means[0]:+.3f}")

    print(f"\n   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Improvement: {losses[0] - losses[-1]:+.4f}")

    print("\n" + "="*80)
