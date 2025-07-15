"""byzAttack: A Flower / PyTorch app."""
import random

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from src.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, attack_info, client_type, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.attack_info = attack_info
        self.client_type = client_type
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        attack_activated = bool(config["attack_activated"])
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.attack_info,
            attack_activated,
            self.client_type,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "ID": self.partition_id},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    attack_info = {"byz-attack-type": context.run_config["byz-attack-type"]}
    if attack_info["byz-attack-type"] == "Gaussian Noise":
        attack_info["mu"] = context.run_config["mu"]
        attack_info["variance"] = context.run_config["variance"]

    random.seed(context.run_config["random-seed"])
    num_malicious = context.run_config["num-malicious"]
    malicious_ids = random.sample(range(num_partitions), num_malicious)

    # Set client type
    if partition_id in malicious_ids:
        client_type = "Malicious"
    else:
        client_type = "Honest"

    # Return Client instance
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs,
        attack_info,
        client_type,
        partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
