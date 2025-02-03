### Define Probe ###

from dataclasses import dataclass
from simple_parsing import Serializable

import torch
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits as bce_with_logits,
    cross_entropy
)


@dataclass
class ProbeConfig(Serializable):
    def to_dict(self):
        irrelevant_fields = []
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}


@dataclass
class LogisticProbeConfig(ProbeConfig):
    l2p: float = 1e-3



PROBE_CONFIGS = {
    "logreg": LogisticProbeConfig
}


class Probe:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def fit(self, acts, labels):
        raise NotImplementedError

    def predict(self, acts):
        raise NotImplementedError

    def filter(self, acts, labels, contamination):
        preds = self.predict(acts)
        disagree = (preds - labels).abs()
        # return indices for bottom (1-contamination) of examples
        return disagree.argsort(descending=True)[int(contamination * len(disagree)):]


class Classifier(torch.nn.Module):
    """Linear classifier trained with supervised learning."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(
            input_dim, num_classes if num_classes > 2 else 1, device=device, dtype=dtype
        )
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        l2_penalty: float = 0.001,
        max_iter: int = 10_000,
    ) -> float:
        """Fits the model to the input data using L-BFGS with L2 regularization.

        Args:
            x: Input tensor of shape (N, D), where N is the number of samples and D is
                the input dimension.
            y: Target tensor of shape (N,) for binary classification or (N, C) for
                multiclass classification, where C is the number of classes.
            l2_penalty: L2 regularization strength.
            max_iter: Maximum number of iterations for the L-BFGS optimizer.

        Returns:
            Final value of the loss function after optimization.
        """
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )

        num_classes = self.linear.out_features
        loss_fn = bce_with_logits if num_classes == 1 else cross_entropy
        loss = torch.inf
        y = y.to(
            torch.get_default_dtype() if num_classes == 1 else torch.long,
        )

        def closure():
            nonlocal loss
            optimizer.zero_grad()

            # Calculate the loss function
            logits = self(x).squeeze(-1)
            loss = loss_fn(logits, y)
            if l2_penalty:
                reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            else:
                reg_loss = loss

            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)
    

class LogisticProbe(Probe):
    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        self.l2p = config.l2p

    def fit(self, acts, labels):
        acts = acts.to(torch.float32)
        self.clf = Classifier(acts.shape[1], num_classes=1, device=acts.device)
        self.clf.fit(acts, labels, l2_penalty=self.l2p)

    def predict(self, acts):
        acts = acts.to(torch.float32)
        preds = torch.sigmoid(self.clf(acts))
        return preds


PROBES = {
    "logreg": LogisticProbe
}