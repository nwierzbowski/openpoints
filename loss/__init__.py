from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .distill_loss import  DistillLoss
from .pointnext_loss import FeatureWeightedChamfer
from .peeler_loss import PeelerLoss
from .build import build_criterion_from_cfg