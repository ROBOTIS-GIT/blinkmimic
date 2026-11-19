import torch

from blinkmimic.sampling_bins import episode_return, probe_distribution, reset_probabilities, update_bin_statistics


def test_repeated_and_masked_episode_bins():
    bins = torch.tensor([0, 0, 1])
    values = torch.tensor([1., 3., 100.])
    stat = torch.zeros(2)
    update_bin_statistics(stat, bins, values, 0.8, mask=torch.tensor([1., 1., 0.]))
    torch.testing.assert_close(stat, torch.tensor([0.4, 0.]))
    result = episode_return(values, values, torch.ones(3), torch.tensor([True, False, True]), 0.5)
    torch.testing.assert_close(result, torch.tensor([1., 6., 100.]))


def test_probe_focus_stops_at_clip_boundary():
    failures = torch.tensor([0., 0., 1., 1., 0.])
    following = torch.tensor([1, 2, 2, 4, 4])
    torch.testing.assert_close(probe_distribution(failures, following, 10), torch.tensor([.25, .25, .25, .25, 0.]))


def test_masked_reset_keeps_effort_when_no_episode_ends():
    effort = torch.tensor([.5, .5])
    weights = reset_probabilities(torch.tensor([1., 0.]), effort, torch.zeros(2),
                                  torch.tensor([0, 0]), 10., .3, torch.tensor(False))
    torch.testing.assert_close(effort, torch.tensor([.5, .5]))
    torch.testing.assert_close(weights.sum(), torch.tensor(1.))
    assert weights[1] > weights[0]


def test_tiny_failure_mass_keeps_the_original_normalization_floor():
    weights = reset_probabilities(torch.zeros(2), torch.full((2,), 0.5),
                                  torch.tensor([0., 1e-7]), torch.tensor([0, 0]), 10., 0.3)
    torch.testing.assert_close(weights, torch.tensor([0.38, 0.35]))
