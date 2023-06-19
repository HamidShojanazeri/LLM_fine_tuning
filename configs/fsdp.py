from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    seed: int=42
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT # alternatively can use FULL_STATE_DICT to assemble the full checkpoints on CPU.
    fsdp_activation_checkpointing: bool=True
    pure_bf16: bool = True
    optimizer: str= "anyprecision"
    
    
    