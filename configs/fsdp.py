from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=True
    seed: int=42
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    fsdp_activation_checkpointing: bool=True
    
    
    