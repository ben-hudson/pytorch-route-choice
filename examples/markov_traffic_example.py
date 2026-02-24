import tntp
import torchdeq
import torch_geometric.utils
import torch
import osmnx
import networkx as nx

from urllib.parse import urljoin
from route_choice import MarkovRouteChoice


def inverse_bpr(free_flow_time, cost, capacity, a, b) -> torch.Tensor:
    # Using torch.relu to handle cases where cost < free_flow_time, which should result in 0 flow.
    base = torch.relu((cost / free_flow_time) - 1.0) / a
    exponent = 1.0 / b
    flow = capacity * torch.pow(base, exponent)
    return flow


class DEQTraffic(torch.nn.Module):

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
            safe_costs = torch.clamp(costs - 0.01 * (supply_flow - demand_flow), min=free_flow_time)
            return safe_costs

        cost_list, info = self.solver(dual_fixed_point_problem, initial_cost)
        return cost_list[-1]


if __name__ == "__main__":
    root = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/refs/heads/master/SiouxFalls/"

    network = tntp.convert_to_networkx(
        tntp.read_node_file(urljoin(root, "SiouxFalls_node.tntp"), index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(urljoin(root, "SiouxFalls_net.tntp"), crs="wgs84"),
        tntp.read_flow_file(urljoin(root, "SiouxFalls_flow.tntp"), u_col="From", v_col="To"),
    )
    demand_table = tntp.read_demand_file(urljoin(root, "SiouxFalls_trips.tntp"))

    node_list = list(network.nodes)
    demand_table = demand_table.reindex(index=node_list, columns=node_list)
    scaling_factor = 1000

    base_graph = torch_geometric.utils.from_networkx(network)

    # here, the "batch" dimension is origin and the "element" dimension is destination
    # this is the way we want it because we want each item in the batch to contain demand from every origin
    demand = torch.as_tensor(demand_table.values.T, dtype=torch.float32) / scaling_factor

    sink_nodes = torch.ones(len(node_list), dtype=torch.long)
    sink_node_mask = torch.diag_embed(sink_nodes)

    route_choice = MarkovRouteChoice(None, f_solver="fixed_point_iter", f_max_iter=10000, f_tol=1e-5, node_dim=-1)
    model = DEQTraffic(route_choice, f_solver="anderson", f_max_iter=5000, f_tol=1e-4, dest_dim=-2)

    costs = model.solve(
        base_graph.edge_index,
        base_graph.free_flow_time.float().unsqueeze(0),
        base_graph.capacity.float().unsqueeze(0) / scaling_factor,
        base_graph.b.float().unsqueeze(0),
        base_graph.power.float().unsqueeze(0),
        demand,
        sink_node_mask,
    )
    costs = costs.squeeze(0)

    # compare computed costs to reference equilibrium costs
    reference_costs = base_graph.Cost.float()
    absolute_error = (costs - reference_costs).abs()
    print(f"Max absolute error: {absolute_error.max().item():.4f}")
    print(f"Mean absolute error: {absolute_error.mean().item():.4f}")
    print(f"Mean relative error: {(absolute_error / reference_costs).mean().item():.4%}")
    assert torch.allclose(costs, reference_costs, atol=1.0, rtol=0.1), "Computed costs differ significantly from reference."

    # plot convergence to fixed point
    converged_cost_dict = {e: cost.item() for e, cost in zip(network.edges, costs)}
    nx.set_edge_attributes(network, converged_cost_dict, "cost")
    edge_color = osmnx.plot.get_edge_colors_by_attr(network, "cost", cmap="RdYlGn_r")
    osmnx.plot.plot_graph(network, edge_color=edge_color, close=True, save=True, filepath="example_parallel.png")
