import math
import torch


def clip_state_update(before, after, max_norm=1.0):
    keys = [k for k in after if k in before]
    sq = torch.tensor(0.0)
    for k in keys:
        d = after[k].float() - before[k].float()
        sq += torch.sum(d*d)
    norm = torch.sqrt(sq).item()
    scale = min(1.0, max_norm / (norm + 1e-12))
    clipped = {}
    for k in keys:
        clipped[k] = before[k] + (after[k] - before[k]) * scale
    return clipped, norm, scale


def add_gaussian_noise(state, sigma, clip_norm):
    if sigma <= 0: return state
    out = {}
    std = sigma * clip_norm
    for k,v in state.items():
        out[k] = v + torch.randn_like(v.float()) * std
    return out


def approximate_epsilon(steps, noise_multiplier, delta=1e-5):
    if noise_multiplier <= 0: return float("inf")
    return (steps ** 0.5) / noise_multiplier * math.sqrt(2 * math.log(1/delta))
