import torch
import tqdm

from markov import MarkovRouteChoice, MarkovRouteChoiceDataset, load_rl_tutorial_network
from sklearn.preprocessing import StandardScaler

# MaxEnt IRL (Ziebart et al., 2008) is almost the same as the Recursive Logit (Fosgerau et al., 2013) model.
# The MDP is defined in terms of a single origin with many possible destinations in MaxEnt IRL.
# This is backwards from how it is defined in the Recursive Logit (i.e. Markovian) model.
# The other difference is the loss: MaxEnt IRL optimizes the expected feature counts.
# For example, the expectation of the travel time (a feature) summed along the route.
# This is a harder task than using the states and transitions directly, so we expect a higher reconstruction loss.
if __name__ == "__main__":
    network = load_rl_tutorial_network().reverse()
    demand = {"d": 1}
    dest = "o"

    reward_fn = lambda e: -2.0 * network.edges[e]["travel_time"] - 0.01
    feat_names = ["travel_time"]

    dataset = MarkovRouteChoiceDataset(
        network,
        feat_names,
        demand={"d": 1},
        dest="o",
        reward_fn=reward_fn,
        n_samples=1024,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    feat_scaler = StandardScaler()
    feats_np = feat_scaler.fit_transform(dataset.graph.edge_attr.numpy())
    feats = torch.as_tensor(feats_np, dtype=torch.float32).unsqueeze(0)

    model = MarkovRouteChoice(torch.nn.Linear(len(feat_names), 1, bias=True))
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=1e-4, threshold_mode="rel", patience=20, min_lr=1e-4
    )

    epochs = tqdm.trange(100)
    for epoch in epochs:
        epoch_loss = 0
        for sources, paths in loader:
            model.train()
            optim.zero_grad()

            reward, value, prob = model(dataset.edge_index, feats, dataset.sink_node_mask)
            node_flows, edge_flows = model.get_flows(dataset.edge_index, prob, dataset.demand)

            # since we oly have one destination here, we can use the edge flows as calculated.
            # otherwise, I think we would have to compare the feature counts per-destination.
            observed_feats = (paths.unsqueeze(-1) * feats).mean()
            pred_feats = (edge_flows.unsqueeze(-1) * feats).mean()
            loss = torch.nn.functional.mse_loss(pred_feats, observed_feats)

            loss.backward()
            optim.step()

            epoch_loss += loss.detach()

        scheduler.step(epoch_loss)
        epochs.set_postfix({"loss": epoch_loss.item(), "lr": scheduler.get_last_lr()[0]})

    with torch.no_grad():
        model.eval()
        observed_flows = dataset.tensors[1].float().mean(dim=0)
        reward, value, prob = model(dataset.edge_index, feats, dataset.sink_node_mask)
        _, pred_flows = model.get_flows(dataset.edge_index, prob, dataset.demand)
        reconstruction_loss = torch.nn.functional.mse_loss(observed_flows, pred_flows.squeeze(0))
        print("Edge flow reconstruction MSE:", reconstruction_loss)
