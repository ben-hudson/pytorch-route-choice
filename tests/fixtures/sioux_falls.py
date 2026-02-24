import pytest
import tntp

from urllib.parse import urljoin


@pytest.fixture
def sioux_falls():
    root = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/refs/heads/master/SiouxFalls/"

    network = tntp.convert_to_networkx(
        tntp.read_node_file(urljoin(root, "SiouxFalls_node.tntp"), index_col="Node", x_col="X", y_col="Y", crs="wgs84"),
        tntp.read_net_file(urljoin(root, "SiouxFalls_net.tntp"), crs="wgs84"),
        tntp.read_flow_file(urljoin(root, "SiouxFalls_flow.tntp"), u_col="From", v_col="To"),
    )
    demand_table = tntp.read_demand_file(urljoin(root, "SiouxFalls_trips.tntp"))

    node_list = list(network.nodes)
    demand_table = demand_table.reindex(index=node_list, columns=node_list)

    return network, demand_table
