"""byzAttack: A Flower / PyTorch app."""
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision import models



from src.attacks import flip_sign, add_gaussian_noise, flip_labels
from src.utils import BackdoorCrossStamp

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

def Net():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=64)
    return trainloader, testloader


def train(net, trainloader, epochs, attack_info, attack_activated, client_type, device, backdoor_label = 3, attack_weight = 2.5):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]

            if attack_activated and client_type == "Malicious":
                match attack_info["byz-attack-type"]:
                    case "Backdoor":
                        stamper = BackdoorCrossStamp()
                        images = stamper.stamp_batch(images)
                        labels = torch.full_like(labels, backdoor_label)
                    case "Label Flip":
                        try:
                            labels = flip_labels(labels, 10)
                        except KeyError:
                            raise KeyError("'num_labels_flipped' must be specified in config file.")

            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            if attack_activated and client_type == "Malicious":
                loss *= attack_weight
            loss.backward()

            if attack_activated and client_type == "Malicious":
                match attack_info["byz-attack-type"]:
                    case "Sign Flip":
                        flip_sign(net.parameters())
                    case "Gaussian Noise":
                        add_gaussian_noise(net.parameters(), attack_info["mu"], attack_info["variance"])

            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def test_attack_efficacy(net, testloader, device, backdoor_label = 3) -> float:
    """Evaluates the efficacy of a backdoor attack on the model.
    Data should be a backdoored dataset. Labels should be the attacker target.
    """
    net.to(device)
    net.eval()
    correct, total = 0, 0

    stamper = BackdoorCrossStamp()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]

            # Filter out samples where the original label is already the backdoor label
            mask = labels != backdoor_label
            if mask.sum() == 0:
                continue

            filtered_images = images[mask]
            filtered_labels = labels[mask]

            attacked_images = stamper.stamp_batch(filtered_images).to(device)
            attacked_labels = torch.full_like(filtered_labels, backdoor_label).to(device)
            
            outputs = net(attacked_images)
            predictions = torch.max(outputs.data, 1)[1]

            correct += (predictions == attacked_labels).sum().item()
            total += attacked_labels.size(0)

    efficacy = correct / total if total > 0 else 0.0
    return efficacy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    # shutil.copy(settings.config_path, save_path)

    return save_path, run_dir


EVAL_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def apply_eval_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [EVAL_TRANSFORMS(img) for img in batch["img"]]
    return batch
