"""Loss plotting utilities for knowledge distillation training.

Adapted from ContextAgent/LLaMA-Factory plotting utilities.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def smooth(scalars: List[float], weight: float = 1.8) -> List[float]:
    """EMA implementation according to TensorBoard.
    
    Args:
        scalars: List of scalar values to smooth
        weight: Smoothing weight (1.8 by default, adjusted via sigmoid)
    
    Returns:
        Smoothed list of the same length
    """
    if len(scalars) == 0:
        return []
    
    last = scalars[0]
    smoothed = []
    # Sigmoid-based adaptive weight based on number of scalars
    adaptive_weight = weight * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)
    
    for next_val in scalars:
        smoothed_val = last * adaptive_weight + (1 - adaptive_weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed


def plot_loss(
    training_log: List[Dict[str, Any]],
    output_path: str,
    keys: Optional[List[str]] = None,
) -> str:
    """Plot loss curves from training log.
    
    Args:
        training_log: List of log entries with 'step' and loss keys
        output_path: Path to save PNG file
        keys: Loss keys to plot (default: ['loss', 'eval_loss'])
    
    Returns:
        Path to saved PNG file
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed. Skipping loss plot.")
        return ""
    
    if keys is None:
        keys = ["loss", "eval_loss"]
    
    # Extract data
    data: Dict[str, List] = {key: [] for key in keys}
    steps = []
    
    for entry in training_log:
        step = entry.get("step", entry.get("epoch", len(steps)))
        if "step" in entry:
            steps.append(step)
            for key in keys:
                if key in entry:
                    data[key].append(entry[key])
    
    if not steps:
        print("No data to plot.")
        return ""
    
    # Create figure
    plt.close("all")
    plt.switch_backend("agg")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each key
    for key in keys:
        if data[key]:
            values = data[key]
            smoothed_values = smooth(values)
            
            ax.plot(steps[:len(values)], values, alpha=0.3, label=f"{key} (raw)")
            ax.plot(steps[:len(smoothed_values)], smoothed_values, linewidth=2, label=f"{key} (smoothed)")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Loss plot saved to {output_path}")
    
    plt.close()
    return output_path


def load_training_log(log_path: str) -> List[Dict[str, Any]]:
    """Load training log from JSON file."""
    with open(log_path, "r") as f:
        return json.load(f)


def plot_from_checkpoint(checkpoint_dir: str, output_filename: str = "loss_curves.png"):
    """Plot loss from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (contains trainer_state.json)
        output_filename: Name of output PNG file
    """
    checkpoint_path = Path(checkpoint_dir)
    log_path = checkpoint_path / "trainer_state.json"
    
    if not log_path.exists():
        print(f"Training log not found at {log_path}")
        return
    
    training_log = load_training_log(str(log_path))
    output_path = checkpoint_path / output_filename
    
    plot_loss(training_log, str(output_path))
