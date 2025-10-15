import torch
import torch_geometric.utils
import tqdm

from purc import dense_incidence_matrix, load_purc_toy_network, PURC
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    network = load_purc_toy_network()
    for n in network.nodes:
        network.nodes[n]["is_orig"] = n == "o"
        network.nodes[n]["is_dest"] = n == "d"

    feat_names = ["rate"]
    torch_graph = torch_geometric.utils.from_networkx(network, group_edge_attrs=feat_names)

    feat_scaler = StandardScaler()
    feats_np = feat_scaler.fit_transform(torch_graph.edge_attr.numpy())
    feats = torch.as_tensor(feats_np, dtype=torch.float32)

    incidence_matrix = dense_incidence_matrix(torch_graph.edge_index, torch_graph.num_nodes, torch_graph.num_edges)

    model = PURC(len(feat_names), regularizer="entropy")
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=1e-4, threshold_mode="rel", patience=10, min_lr=1e-4
    )

    epochs = tqdm.trange(100)
    for epoch in epochs:
        model.train()
        optim.zero_grad()

        _, loss = model(incidence_matrix, torch_graph.length, feats.unsqueeze(0), torch_graph.flow.unsqueeze(0))

        loss.backward()
        optim.step()
        scheduler.step(loss)

        epochs.set_postfix({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]})

    with torch.no_grad():
        model.eval()
        util_rates = model.util_rate(feats)
        print("Edge util rates (relative):", util_rates.squeeze())
        print("Ground truth rates:", torch_graph.edge_attr.squeeze())
