from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardedStateDictConfig,
    StateDictType,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT01_ROOT = PROJECT_ROOT.parent / "01_end_to_end_lm"
if str(PROJECT01_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT01_ROOT))

from student_lm import AdamW, GPTConfig, TransformerLM  # noqa: E402
from student_systems import ddp_train_step  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("this FSDP exercise requires Linux/NVIDIA GPUs")
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=2")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(336)

    config = GPTConfig(vocab_size=128, d_model=128, num_heads=8, num_layers=4, d_ff=352)
    model = FullyShardedDataParallel(
        TransformerLM(config).to(device),
        device_id=device,
        use_orig_params=True,
    )
    optimizer = AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(2000 + rank)
    for _ in range(3):
        inputs = torch.randint(0, config.vocab_size, (4, 64), generator=generator, device=device)
        targets = torch.roll(inputs, shifts=-1, dims=1)
        ddp_train_step(model, inputs, targets, optimizer)

    output = PROJECT_ROOT / "artifacts" / "fsdp_checkpoint"
    output.mkdir(parents=True, exist_ok=True)
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
    ):
        torch.save(model.state_dict(), output / f"model-rank-{rank:02d}.pt")
    torch.save(optimizer.state_dict(), output / f"optimizer-rank-{rank:02d}.pt")
    dist.barrier()
    if rank == 0:
        print(f"FSDP smoke passed; sharded checkpoint: {output}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
