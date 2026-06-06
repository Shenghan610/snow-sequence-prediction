import argparse
import copy
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


SCALAR_FEATURE_DIM = 11
SEASON_FEATURE_DIM = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Train the snow-cover sequence predictor.")
    # Default config for the next PyCharm run:
    # restore703 backbone + multi-baseline gated residual head.
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--seq-len", type=int, default=21)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--base-window", type=int, default=7)
    parser.add_argument("--max-delta", type=float, default=0.40)
    parser.add_argument("--hidden-dropout", type=float, default=0.10)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--head-dropout", type=float, default=0.10)
    parser.add_argument("--high-snow-weight", type=float, default=2.0)
    parser.add_argument("--change-weight", type=float, default=1.2)
    parser.add_argument("--energy-weight", type=float, default=0.001)
    parser.add_argument("--anchor-weight", type=float, default=0.02)
    parser.add_argument("--residual-l1-weight", type=float, default=0.003)
    parser.add_argument("--residual-gate-bias", type=float, default=-0.8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta-r2", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--no-ema", action="store_true", default=True)
    parser.add_argument("--use-ema", action="store_false", dest="no_ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def load_model_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
