import torch

from samplers import GaussianSampler
from tasks import LinearRegression, QuadraticRegression


def test_gaussian_sampler_is_seeded():
    sampler = GaussianSampler(n_dims=2)
    sample_a = sampler.sample_xs(n_points=3, b_size=2, seeds=[1, 2])
    sample_b = sampler.sample_xs(n_points=3, b_size=2, seeds=[1, 2])
    assert torch.equal(sample_a, sample_b)


def test_quadratic_regression_matches_squared_input_definition():
    task = QuadraticRegression(n_dims=2, batch_size=1, seeds=[0])
    xs = torch.tensor([[[2.0, 3.0]]])
    ys = task.evaluate(xs)
    manual = ((xs**2) @ task.w_b)[:, :, 0] / (3**0.5)
    assert torch.allclose(ys, manual)


def test_linear_regression_uses_seeded_weights():
    task_a = LinearRegression(n_dims=2, batch_size=1, seeds=[5])
    task_b = LinearRegression(n_dims=2, batch_size=1, seeds=[5])
    assert torch.equal(task_a.w_b, task_b.w_b)
