import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.task import test_attack_efficacy


# Dummy model that predicts class 0 (attack success) if x ends in 2, else class 1
class DummyModel(nn.Module):
    def forward(self, x):
        # Input shape: [batch_size, 1]
        x = x.squeeze(1).long()  # Convert shape [batch_size, 1] -> [batch_size]
        pred = (x % 10 != 2).long()  # 0 if ends in 2, else 1
        return F.one_hot(pred, num_classes=2).float()


# Dummy stamper that does nothing
class DummyStamper:
    def stamp_batch(self, x):
        return x  # No change


def test_efficacy():
    n = 10_000
    x = torch.arange(n).unsqueeze(1).float()  # Shape: [n, 1]
    y = torch.ones(n, dtype=torch.long)       # All labels are 1 (not the backdoor label)

    # Prepare dataset in the expected dict format
    dataset = [{"img": xi, "label": yi} for xi, yi in zip(x, y)]
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=lambda b: {
        "img": torch.stack([item["img"] for item in b]),
        "label": torch.tensor([item["label"] for item in b]),
    })

    # Patch the stamper in src.task
    import src.task
    src.task.BackdoorCrossStamp = DummyStamper  # Replace with dummy

    model = DummyModel()
    device = "cpu"
    stamp_config = {"backdoor_label": 0}

    efficacy = test_attack_efficacy(model, dataloader, device, stamp_config)

    # Expect 1/10 attack success rate (only numbers ending in 2 succeed)
    expected = 0.1
    if not abs(efficacy - expected) < 1e-3:
        raise ValueError(f"Efficacy should be {expected}, got {efficacy:.4f}")
    print("✅ All tests passed!")

def test_efficacy_original_labels_are_equal_to_backdoor():
    n = 10_000
    x = torch.arange(n).unsqueeze(1).float()  # Shape: [n, 1]
    y = torch.zeros(n, dtype=torch.long)       # All labels are 0 (the backdoor label)

    # Prepare dataset in the expected dict format
    dataset = [{"img": xi, "label": yi} for xi, yi in zip(x, y)]
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=lambda b: {
        "img": torch.stack([item["img"] for item in b]),
        "label": torch.tensor([item["label"] for item in b]),
    })

    # Patch the stamper in src.task
    import src.task
    src.task.BackdoorCrossStamp = DummyStamper  # Replace with dummy

    model = DummyModel()
    device = "cpu"
    stamp_config = {"backdoor_label": 0}

    efficacy = test_attack_efficacy(model, dataloader, device, stamp_config)

    # Expect 0. attack success rate (only numbers ending in 2 succeed)
    if efficacy != 0.:
        raise ValueError(f"Efficacy should be 0., got {efficacy:.4f}")
    print("✅ All tests passed!")

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.task import test_attack_efficacy


# Modelo que siempre predice clase 1 (nunca la clase backdoor 0)
class AlwaysWrongModel(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return F.one_hot(torch.ones(batch_size, dtype=torch.long), num_classes=2).float()


# Dummy stamper sin efecto
class DummyStamper:
    def stamp_batch(self, x):
        return x  # No altera la entrada


def test_efficacy_zero_success():
    n = 1_000
    x = torch.arange(n).unsqueeze(1).float()
    y = torch.ones(n, dtype=torch.long)

    dataset = [{"img": xi, "label": yi} for xi, yi in zip(x, y)]
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=lambda b: {
        "img": torch.stack([item["img"] for item in b]),
        "label": torch.tensor([item["label"] for item in b]),
    })

    # Mockeo del stamper
    import src.task
    src.task.BackdoorCrossStamp = DummyStamper

    model = AlwaysWrongModel()
    device = "cpu"
    stamp_config = {"backdoor_label": 0}

    efficacy = test_attack_efficacy(model, dataloader, device, stamp_config)

    if efficacy != 0.0:
        raise ValueError(f"Efficacy should be 0.0, got {efficacy:.4f}")
    print("✅ Zero-success test passed!")



if __name__ == "__main__":
    test_efficacy()
    test_efficacy_original_labels_are_equal_to_backdoor()
    test_efficacy_zero_success()
