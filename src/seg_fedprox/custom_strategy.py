from typing import List, Optional, Tuple

from flwr.common import FitIns, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from configs.fedprox_config import *
from src.seg_fedprox.model import UNetModel, dice_coef, dice_coef_loss, iou


class SaveModelStrategy(FedAvg):
    def __init__(self, mu=0.1, num_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.num_rounds = num_rounds  # Total number of rounds
        self.final_parameters: Optional[Parameters] = None  # Final global weights

    def configure_fit(self, server_round, parameters, client_manager):
        # Your existing FedProx configuration
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        new_instructions = []
        for client_proxy, fit_ins in client_instructions:
            new_config = dict(fit_ins.config)
            new_config["mu"] = self.mu
            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=new_config)
            new_instructions.append((client_proxy, new_fit_ins))
        return new_instructions

    def aggregate_fit(self, server_round, results, failures):
        # First aggregate normally
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            model = UNetModel().model
            model.compile(
                optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef, iou]
            )
            model.set_weights(weights)
            # Save with round number in filename
            filename = (
                INITIAL_FED_PATH
                + PROBLEM_TYPE
                + f"global_model_round_{server_round}.h5"
            )
            model.save(filename)

        # Save final parameters after last round
        if server_round == self.num_rounds:
            print(f"Saving final parameters after round {server_round}")
            self.final_parameters = aggregated_parameters

        return aggregated_parameters, aggregated_metrics
