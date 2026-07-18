import torch

from llm_from_scratch.moe import TopKMoE


def main() -> None:
    torch.manual_seed(7)
    moe = TopKMoE(32, 64, num_experts=8, top_k=2, capacity_factor=1.0, shared_expert=True)
    x = torch.randn(4, 16, 32)
    output, stats = moe(x)
    print("output:", tuple(output.shape))
    print("capacity:", stats["capacity"].item())
    print("selected load:", stats["selected_load"].tolist())
    print("accepted load:", stats["accepted_load"].tolist())
    print("dropped assignments:", stats["dropped"].sum().item())
    print("balance loss:", stats["balance_loss"].item())
    print("z-loss:", stats["z_loss"].item())


if __name__ == "__main__":
    main()

