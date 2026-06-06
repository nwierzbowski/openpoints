from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .distill_loss import  DistillLoss
from .pointnext_loss import FeatureWeightedChamfer
from .build import build_criterion_from_cfg