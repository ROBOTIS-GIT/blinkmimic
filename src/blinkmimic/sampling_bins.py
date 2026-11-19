"""Tensor operations shared by BlinkMimic reset samplers across physics backends."""

import torch


def episode_return(discounted, total, length, terminated, gamma):
    """Bootstrap truncated episodes using their mean step reward."""
    bootstrap = (gamma**length) * (total / length) / (1 - gamma)
    return discounted + torch.where(terminated, 0.0, bootstrap)


def update_bin_statistics(stat, bins, values, old, *, average=True, mask=None):
    """Apply one EMA per sampled bin, optionally using a fixed-size episode mask."""
    if mask is None and not average:
        stat.index_put_((bins,), old * stat[bins] + (1 - old) * values)
        return
    weights = torch.ones_like(values) if mask is None else mask
    sums = torch.zeros_like(stat).index_add_(0, bins, values if mask is None else values * weights)
    counts = torch.zeros_like(stat).index_add_(0, bins, weights)
    stat.copy_(torch.where(counts > 0, old * stat + (1 - old) * sums / counts.clamp(min=1), stat))


def reset_probabilities(returns, effort, failures, prev, max_return, failure_share, gate=None):
    """Advance the effort estimate and compute reset weights before optional probe focus."""
    soft = torch.softmax(-25.0 * returns / max_return, 0)
    if gate is None:
        effort.mul_(0.999).add_(0.001 * soft)
    else:
        effort.copy_(torch.where(gate, effort * 0.999 + 0.001 * soft, effort))
    probs = 0.5 / returns.numel() + 0.4 * soft + 0.1 * effort / effort.sum()
    if failure_share > 0:
        fail = torch.zeros_like(failures).index_add_(0, prev, failures)
        fail = fail / fail.sum().clamp(min=1e-6)
        probs = (1 - failure_share) * probs + failure_share * torch.where(fail.sum() > 0, fail, probs)
    return probs


def probe_distribution(failures, next_bin, window):
    """Spread each failure over preceding bins within its clip; requires positive mass."""
    weights = failures.clone()
    focus = weights.clone()
    has_next = next_bin != torch.arange(len(next_bin), device=next_bin.device)
    for _ in range(window):
        weights = weights[next_bin] * has_next
        focus += weights
    return focus / focus.sum()


def record_probe_failures(counts, bins, decay):
    counts.mul_(decay)
    counts.index_add_(0, bins, torch.ones_like(bins, dtype=counts.dtype))
