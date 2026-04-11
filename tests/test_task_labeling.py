import pytest
import torch
from munch import Munch

from task_labeling import TaskLabeler


def test_task_labeler_appends_manual_label_values():
    config = Munch(
        enabled=True,
        dimension=2,
        mode="manual",
        manual_map={"linear": [0.0, 1.0]},
    )
    labeler = TaskLabeler(config, base_n_dims=2)
    xs = torch.zeros(2, 3, 4)
    labeled = labeler.apply(xs, "linear")
    assert labeled[:, :, 2:].tolist() == [[[0.0, 1.0]] * 3] * 2


def test_task_labeler_rejects_missing_manual_mapping():
    config = Munch(enabled=True, dimension=1, mode="manual", manual_map={})
    labeler = TaskLabeler(config, base_n_dims=2)
    with pytest.raises(KeyError):
        labeler.apply(torch.zeros(1, 1, 3), "quadratic")
