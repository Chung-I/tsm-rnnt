import torch
from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer

Registrable._registry[Optimizer]["adamw"] = torch.optim.AdamW
