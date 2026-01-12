import torch
import torchdeq


def inverse_bpr(
    free_flow_time: torch.Tensor, cost: torch.Tensor, capacity: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    # Using torch.relu to handle cases where cost < free_flow_time, which should result in 0 flow.
    base = torch.relu((cost / free_flow_time) - 1.0) / a
    exponent = 1.0 / b
    flow = capacity * torch.pow(base, exponent)
    return flow


class MarkovTrafficEquilibrium(torch.nn.Module):
    def __init__(self, route_choice_model: torch.nn.Module, dest_dim: int = -1, **solver_kwargs):
        super().__init__()
        self.route_choice_model = route_choice_model
        assert dest_dim < 0, "node_dim must be specified as a negative offset."
        self.dest_dim = dest_dim
        self.solver = torchdeq.get_deq(**solver_kwargs)

    def solve(
        self,
        edge_index: torch.Tensor,
        free_flow_time: torch.Tensor,
        capacity: torch.Tensor,
        b: torch.Tensor,
        power: torch.Tensor,
        demand: torch.Tensor,
        sink_node_mask: torch.Tensor,
        initial_cost: torch.Tensor = None,
    ):
        # initial link costs as free_flow travel time
        if initial_cost is None:
            initial_cost = free_flow_time.clone()

        def dual_fixed_point_problem(costs):
            supply_flow = inverse_bpr(free_flow_time, costs, capacity, b, power)

            costs_shape = list(costs.shape)
            costs_shape[self.dest_dim] = demand.size(self.dest_dim)
            rewards = -costs.expand(*costs_shape)

            _, probs = self.route_choice_model.get_values_and_probs(edge_index, -costs, sink_node_mask)
            _, edge_flows = self.route_choice_model.get_flows(edge_index, probs, demand)
            demand_flow = edge_flows.sum(dim=self.dest_dim, keepdim=True)

            # safeguarding PUMDP value iterations with non-negative costs
            safe_costs = torch.relu(costs - 0.1 * (supply_flow - demand_flow))
            return safe_costs

        cost_list, info = self.solver(dual_fixed_point_problem, initial_cost)
        return cost_list[-1]
