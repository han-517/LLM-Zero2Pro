from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT01_ROOT = PROJECT_ROOT.parent / "01_end_to_end_lm"
if str(PROJECT01_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT01_ROOT))

from student_lm import AdamW, GPTConfig, TransformerLM  # noqa: E402
from student_systems import ddp_train_step  # noqa: E402


def main() -> int:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=2")
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    torch.manual_seed(336)
    config = GPTConfig(vocab_size=32, d_model=16, num_heads=2, num_layers=1, d_ff=32)
    model = TransformerLM(config).to(device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
    optimizer = AdamW(ddp.parameters(), lr=1e-2, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    inputs = torch.randint(0, config.vocab_size, (4, 16), generator=generator, device=device)
    targets = torch.roll(inputs, shifts=-1, dims=1)
    loss = ddp_train_step(ddp, inputs, targets, optimizer)

    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in ddp.parameters()]
    ).sum()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError("DDP parameters diverged across ranks")
    if rank == 0:
        print(f"DDP smoke passed: world_size={dist.get_world_size()}, loss={float(loss):.6f}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
