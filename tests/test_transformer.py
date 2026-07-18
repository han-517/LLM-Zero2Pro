import pytest
import torch

from llm_from_scratch.transformer import GPTConfig, RMSNorm, TinyGPT


def tiny_model() -> TinyGPT:
    return TinyGPT(
        GPTConfig(vocab_size=13, block_size=8, n_layer=2, n_head=4, n_kv_head=2, d_model=16)
    )


def test_rmsnorm_matches_formula() -> None:
    module = RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(module(x), expected)


def test_tiny_gpt_shapes_loss_and_gradients() -> None:
    torch.manual_seed(10)
    model = tiny_model()
    tokens = torch.randint(0, 13, (3, 6))
    targets = torch.randint(0, 13, (3, 6))
    logits, loss, caches = model(tokens, targets)
    assert logits.shape == (3, 6, 13)
    assert loss is not None and loss.ndim == 0
    assert len(caches) == 2
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_tiny_gpt_cache_matches_full_forward() -> None:
    torch.manual_seed(11)
    model = tiny_model().eval()
    tokens = torch.randint(0, 13, (1, 6))
    full, _, _ = model(tokens)
    caches = None
    pieces = []
    for index in range(tokens.shape[1]):
        current, _, caches = model(tokens[:, index : index + 1], caches=caches)
        pieces.append(current)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)


def test_tiny_gpt_can_overfit_one_batch() -> None:
    torch.manual_seed(12)
    model = TinyGPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=2, d_model=8))
    tokens = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    targets = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    with torch.no_grad():
        _, initial_loss, _ = model(tokens, targets)
    for _ in range(35):
        _, loss, _ = model(tokens, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, final_loss, _ = model(tokens, targets)
    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss * 0.35


def test_cached_and_uncached_greedy_generation_match_and_restore_mode() -> None:
    model = tiny_model()
    model.train()
    prefix = torch.tensor([[1, 2]])
    cached = model.generate(prefix, 3, temperature=0, use_cache=True)
    assert model.training
    uncached = model.generate(prefix, 3, temperature=0, use_cache=False)
    assert model.training
    assert torch.equal(cached, uncached)
    assert cached.shape == (1, 5)


def test_generation_rejects_invalid_sampling_arguments() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="top_k"):
        model.generate(torch.tensor([[1]]), 1, top_k=0)

