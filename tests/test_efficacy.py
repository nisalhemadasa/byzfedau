import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from src.utils import evaluate_efficacy


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """If input ends with 2 returns 0 else 1"""
        return F.one_hot((x % 10 != 2).long(), num_classes=2)


def test_efficacy():
    """Dummy test for efficacy evaluation also knonw as ASR (Attackt success rate)."""
    n = 10_000
    x = torch.arange(n)
    y = torch.ones(n, dtype=torch.long)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=128)

    model = DummyModel()
    efficacy = evaluate_efficacy(model, dataloader)
    if efficacy != 0.9:
        raise ValueError(f"Efficacy should be 0.9. Value: {efficacy}")
    print("all test passed :)")


if __name__ == "__main__":
    test_efficacy()
