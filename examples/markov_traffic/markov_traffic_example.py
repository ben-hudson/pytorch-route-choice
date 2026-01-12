import pathlib
import tntp  # pip install git+https://github.com/ben-hudson/pytntp
import torch
import torch_geometric.utils

from route_choice import MarkovRouteChoice
from mte import MarkovTrafficEquilibrium


if __name__ == "__main__":
    root = pathlib.Path("../pytntp/data/SiouxFalls")
    assert root.exists(), "Download the Sioux Falls network from https://github.com/bstabler/TransportationNetworks"
    network = tntp.convert_to_networkx(
        tntp.read_node_file(root / "SiouxFalls_node.tntp", index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(root / "SiouxFalls_net.tntp", crs="wgs84"),
        tntp.read_flow_file(root / "SiouxFalls_flow.tntp", u_col="From", v_col="To"),
    )

    node_list = list(network.nodes)
    demand_table = tntp.read_demand_file(root / "SiouxFalls_trips.tntp").reindex(index=node_list, columns=node_list)
    scaling_factor = 1000

    base_graph = torch_geometric.utils.from_networkx(network)

    # here, the "batch" dimension is origin and the "element" dimension is destination
    # this is the way we want it because we want each item in the batch to contain demand from every origin
    demand = torch.as_tensor(demand_table.values, dtype=torch.float32) / scaling_factor

    sink_nodes = torch.ones(len(node_list), dtype=torch.long)
    sink_node_mask = torch.diag_embed(sink_nodes).T  # transpose because we want it to be orig x dest

    route_choice = MarkovRouteChoice(None, f_solver="fixed_point_iter", f_max_iter=10000, f_tol=1e-5, node_dim=-1)
    model = MarkovTrafficEquilibrium(route_choice, f_solver="anderson", f_max_iter=1000, f_tol=1e-4, dest_dim=-2)

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
    loss = torch.nn.functional.mse_loss(costs, base_graph.Cost)
