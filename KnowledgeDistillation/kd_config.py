"""Global configuration for Knowledge Distillation.

Manages sandbox/live mode similar to ContextAgent pattern.
"""

import os

# Recommended names
MODES = {"live", "sandbox"}
DEFAULT_MODE = "live"

# Priority: ENV -> default
MODE = os.getenv("APP_MODE", DEFAULT_MODE).lower()
if MODE not in MODES:
    MODE = DEFAULT_MODE


def set_mode(mode: str) -> None:
    """Set global mode at runtime (used by argparse)."""
    m = (mode or "").lower()
    if m not in MODES:
        raise ValueError(f"Unknown mode '{mode}', must be one of {sorted(MODES)}")
    global MODE
    MODE = m


def is_sandbox() -> bool:
    """Check if running in sandbox mode (with dummy data/models)."""
    return MODE != "live"
