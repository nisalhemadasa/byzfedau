import json
from logging import INFO, WARNING
from typing import Optional, Union

import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace, aggregate
import wandb

from src.task import set_weights, create_run_dir

PROJECT_NAME = "FLTA-Project"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality."""

    def __init__(self, model_architecture, attack_type=None, use_wandb=False, **kwargs):
        super().__init__(**kwargs)
        # A dictionary to store client gradients as they come
        self.gradients = {}
        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir()
        # Initialise W&B if set
        self.use_wandb = use_wandb
        self.attack_type = attack_type
        self.model = model_architecture
        if self.use_wandb:
            self._init_wandb_project()
        # Keep track of best acc
        self.best_acc_so_far = 0.0
        # Keep track of best loss
        self.best_loss_so_far = None
        self.initial_loss = None
        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        if self.attack_type is not None:
            match self.attack_type:
                case "Label Flip":
                    name = (
                        f"{str(self.run_dir)}-Label Flip"
                    )
                case "Sign Flip":
                    name = (
                        f"{str(self.run_dir)}-Sign Flip"
                    )
                case "Gaussian Noise":
                    name = (
                        f"{str(self.run_dir)}-Gaussian Noise"
                    )
                case "BackDoor":
                    name = (
                        f"{str(self.run_dir)}-BackDoor"
                    )
                case _:
                    raise ValueError(f"Invalid attack type: {self.attack_type}")
            wandb.init(project=PROJECT_NAME, name=name)
        else:
            wandb.init(
                project=PROJECT_NAME,
                name=f"{str(self.run_dir)}-No attack",
            )

    def _store_results(self, tag: str, results_dict) -> None:
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, server_round: int, accuracy, parameters: Parameters) -> None:
        """
        Determines if a new best global model has been found. If so, the model checkpoint is saved to disk.
        :param server_round: current server round.
        :param accuracy: the accuracy of the global model.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead, we are going to apply them to a PyTorch model and save the state dict.
            model = self.model
            set_weights(model, parameters_to_ndarrays(parameters))
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{server_round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def _store_results_and_log(self, server_round: int, tag: str, results_dict) -> None:
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round: int, parameters: Parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Save loss if new best central loss is found
        if self.best_loss_so_far is None or (self.best_loss_so_far is not None and loss <= self.best_loss_so_far):
            self.best_loss_so_far = loss
            log(INFO, "💡 New best global loss found: %f", loss)

        # Store and log
        self._store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Store the gradients of all successfully received clients.
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics["ID"]
            tensor_list = [param.tolist() for param in parameters_to_ndarrays(fit_res.parameters)]
            if client_id not in self.gradients.keys():
                self.gradients[client_id] = [tensor_list]
            else:
                self.gradients[client_id].append(tensor_list)
        self._store_results_and_log(
            server_round=server_round,
            tag="gradients",
            results_dict={"grad_dict": self.gradients},
        )

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
