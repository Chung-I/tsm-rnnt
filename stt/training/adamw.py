import torch
import apex
from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer

Registrable._registry[Optimizer]["adamw"] = torch.optim.AdamW
Registrable._registry[Optimizer]["fused_adam"] = apex.optimizers.FusedAdam
