"""byzAttack: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.task import Net, get_weights


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


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    on_fit_config_scheduler = OnFitConfigScheduler(activation_round=context.run_config["attack-activation-round"])

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config_scheduler.on_fit_config,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
