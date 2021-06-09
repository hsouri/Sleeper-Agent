"""Main class, holding information about models and training/testing routines."""

import torch
from ..utils import bypass_last_layer
from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchBullsEye(_Witch):
    """Brew poison frogs variant with averaged feature matching instead of sums of feature matches.

    This is also known as BullsEye Polytope Attack.

    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, true_classes):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Carve up the model
            feature_model, last_layer = bypass_last_layer(model)

            # Get standard output:
            outputs = feature_model(inputs)
            outputs_sources = feature_model(sources)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()

            feature_loss = (outputs.mean(dim=0) - outputs_sources.mean(dim=0)).pow(2).mean()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure
