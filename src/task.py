"""byzAttack: A Flower / PyTorch app."""
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from src.attacks import flip_sign, add_gaussian_noise, flip_labels, BackdoorCrossStamp
from src.nn import MNISTNet, CifarNet

def get_nn(dataset: str = "MINST") -> torch.nn.Module:
    if dataset == "MNIST":
        return MNISTNet()
    if dataset == "CIFAR10":
        return CifarNet()
    return None

fds = None  # Cache FederatedDataset


def load_dataset_repo(dataset: str = "MNIST"):
    if dataset == "MNIST":
        return "uoft-cs/mnist"
    if dataset == "CIFAR10":
        return "uoft-cs/cifar10"


def load_data(partition_id: int, num_partitions: int, dataset: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=load_dataset_repo(dataset),
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    class DynamicNormalize:
        def __init__(self, mean=0.5, std=0.5):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            c = tensor.size(0)
            mean = [self.mean] * c
            std = [self.std] * c
            return Normalize(tensor, mean, std)

    pytorch_transforms = Compose([
        ToTensor(),
        DynamicNormalize(0.5, 0.5)
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=64)
    return trainloader, testloader


def train(net, trainloader, epochs, attack_info, attack_activated, client_type, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    
    if attack_activated and client_type == "Malicious":
        stamper = BackdoorCrossStamp()
        backdoor_label = attack_info["backdoor_label"]
        
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]

            if attack_activated and client_type == "Malicious":
                match attack_info["byz-attack-type"]:
                    case "Backdoor":
                        images = stamper.stamp_batch(images)
                        labels = torch.full_like(labels, backdoor_label)
                    case "Label Flip":
                        try:
                            labels = flip_labels(labels, 10)
                        except KeyError:
                            raise KeyError("'num_labels_flipped' must be specified in config file.")

            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
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


def test_attack_efficacy(net, testloader, device, attack_info: dict) -> float:
    """
    Evaluates the efficacy of a backdoor attack on the model.
    Ignores samples that already have the backdoor label.
    """
    net.to(device)
    net.eval()
    total, correct = 0, 0

    backdoor_label = attack_info["backdoor_label"]
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