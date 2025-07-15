from logging import WARNING
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


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A dictionary to store client gradients as they come
        self.gradients = {}

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
            tensor_list = [torch.from_numpy(param) for param in parameters_to_ndarrays(fit_res.parameters)]
            if client_id not in self.gradients.keys():
                self.gradients[client_id] = [tensor_list]
            else:
                self.gradients[client_id].append(tensor_list)
                print(f"Length : {len(self.gradients[client_id])}")



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
