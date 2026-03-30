import matplotlib.pyplot as plt
import networkx as nx
import osmnx
import tntp
import torch
import torch_geometric.utils
import torchdeq

from route_choice import MarkovRouteChoice
from urllib.parse import urljoin


def bpr(free_flow_time, flow, capacity, b, power) -> torch.Tensor:
    """Bureau of Public Roads link cost function."""
    return free_flow_time * (1.0 + b * torch.pow(flow / capacity, power))


class DEQTraffic(torch.nn.Module):
    def __init__(
        self,
        route_choice_model: torch.nn.Module,
        dest_dim: int = -1,
        step_size: float = 1.0,
        **solver_kwargs,
    ):
        super().__init__()
        self.route_choice_model = route_choice_model
        assert dest_dim < 0, "node_dim must be specified as a negative offset."
        self.dest_dim = dest_dim
        assert 0.0 <= step_size <= 1.0, "step_size must be between 0 and 1."
        self.step_size = step_size
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
            _, probs, _ = self.route_choice_model.get_values_and_probs(edge_index, -costs, sink_node_mask)
            _, edge_flows, _ = self.route_choice_model.get_flows(edge_index, probs, demand)
            demand_flow = edge_flows.sum(dim=self.dest_dim, keepdim=True)

            new_costs = bpr(free_flow_time, demand_flow, capacity, b, power)
            return self.step_size * new_costs + (1 - self.step_size) * costs

        cost_list, info = self.solver(dual_fixed_point_problem, initial_cost)
        return cost_list[-1], info


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

    base_graph = torch_geometric.utils.from_networkx(network)

    demand = torch.as_tensor(demand_table.values.T, dtype=torch.float32)

    sink_nodes = torch.ones(len(node_list), dtype=torch.long)
    sink_node_mask = torch.diag_embed(sink_nodes)

    route_choice = MarkovRouteChoice(None, f_solver="anderson", f_max_iter=100, f_tol=1e-5, node_dim=-1)
    model = DEQTraffic(route_choice, f_solver="anderson", f_max_iter=1000, f_tol=1e-4, dest_dim=-2, step_size=0.1)

    print("Solving for equilibrium link costs...")
    scaling_factor = 1000
    costs, info = model.solve(
        base_graph.edge_index,
        base_graph.free_flow_time.float().unsqueeze(0),
        base_graph.capacity.float().unsqueeze(0) / scaling_factor,
        base_graph.b.float().unsqueeze(0),
        base_graph.power.float().unsqueeze(0),
        demand / scaling_factor,
        sink_node_mask,
    )
    costs = costs.squeeze(0)

    reference_costs = base_graph.Cost.float()
    absolute_error = (costs - reference_costs).abs()
    relative_error = absolute_error / reference_costs
    print(f"\nComparison to reference equilibrium costs:")
    print(f"  Mean absolute error: {absolute_error.mean().item():.4f}")
    print(f"  Max absolute error:  {absolute_error.max().item():.4f}")
    print(f"  Mean relative error: {relative_error.mean().item():.2%}")

    # Plot convergence and equilibrium costs side by side
    fig, axes = plt.subplots(1, 2)

    nstep = int(info["nstep"].max().item())
    abs_trace = info["abs_trace"].squeeze(0)[:nstep]
    axes[1].semilogy(abs_trace.numpy())
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Absolute residual")
    axes[1].set_title("Solver convergence")

    nx.set_edge_attributes(network, dict(zip(network.edges, costs.tolist())), "cost")
    edge_color = osmnx.plot.get_edge_colors_by_attr(network, "cost", cmap="RdYlGn_r")
    axes[0].set_facecolor("k")
    osmnx.plot.plot_graph(network, ax=axes[0], edge_color=edge_color, show=False)
    axes[0].set_title("Equilibrium link costs")

    plt.tight_layout()
    plt.show()
