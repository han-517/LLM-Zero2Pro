import torch

from checks.exercises._loader import load_starter

student = load_starter("19_posttraining.py")


def test_sequence_logprob_masks_prompt_and_padding() -> None:
    logp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    torch.testing.assert_close(student.sequence_logprob(logp, mask), torch.tensor([-5.0]))


def test_constant_reward_group_has_zero_advantage() -> None:
    rewards = torch.tensor([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
    advantages = student.group_advantages(rewards)
    torch.testing.assert_close(advantages[0], torch.zeros(3))
    torch.testing.assert_close(advantages[1].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
