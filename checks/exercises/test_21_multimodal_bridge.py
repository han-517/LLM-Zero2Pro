import pytest
import torch

from checks.exercises._loader import load_starter

student = load_starter("21_multimodal_bridge.py")


def test_patchify_shape_and_order() -> None:
    image = torch.arange(16.0).reshape(1, 1, 4, 4)
    patches = student.patchify(image, 2)
    assert patches.shape == (1, 4, 4)
    torch.testing.assert_close(patches[0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_projection_matches_linear_algebra() -> None:
    patches = torch.ones(1, 2, 3)
    weight = torch.ones(3, 4)
    out = student.project_vision(patches, weight, torch.arange(4.0))
    torch.testing.assert_close(out, torch.tensor([[[3.0, 4.0, 5.0, 6.0]]]).expand(1, 2, 4))
    with pytest.raises(ValueError):
        student.patchify(torch.zeros(1, 3, 5, 4), 2)
