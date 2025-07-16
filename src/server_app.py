"""byzAttack: A Flower / PyTorch app."""
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader
import torch

from src.custom_strategy import CustomFedAvg
from src.task import Net, get_weights, set_weights, test, apply_eval_transforms


class OnFitConfigScheduler:
    def __init__(self, activation_round):
        self.activation_round = activation_round
        self.attack_activated = False

    def on_fit_config(self, server_round: int):
        """
        Construct `config` that clients receive when running `fit()`
        :param server_round: server round
        """
        if self.activation_round != 0 and server_round >= self.activation_round:
            self.attack_activated = True

        return {
            "attack_activated": self.attack_activated,
        }

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device = "cpu",
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    strategy_name = context.run_config["strategy-name"]
    on_fit_config_scheduler = OnFitConfigScheduler(activation_round=context.run_config["attack-activation-round"])

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    global_test_set = load_dataset("uoft-cs/cifar10")["test"]

    testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=32,
    )

    strategy_args = {
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 1.0,
        "initial_parameters": parameters,
        "on_fit_config_fn": on_fit_config_scheduler.on_fit_config,
        "min_available_clients": 2,
        "evaluate_fn": gen_evaluate_fn(testloader),
    }

    match strategy_name:
        case "Custom-FedAvg":
            extra_args = {
                "model_architecture": Net(),
                "use_wandb": context.run_config["use-wandb"],
                "attack_type": context.run_config["byz-attack-type"]
            }
            strategy_args.update(extra_args)
            strategy = CustomFedAvg(**strategy_args)
        case _:
            strategy = FedAvg(**strategy_args)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
