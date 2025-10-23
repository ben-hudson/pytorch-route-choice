import torch
import tqdm

from route_choice import MarkovRouteChoice, MarkovRouteChoiceDataset, load_rl_tutorial_network
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    network = load_rl_tutorial_network()
    reward_fn = lambda e: -2.0 * network.edges[e]["travel_time"] - 0.01
    feat_names = ["travel_time"]
    dataset = MarkovRouteChoiceDataset(
        network,
        feat_names,
        demand={"o": 1},
        dest="d",
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

            reward, value, prob = model(dataset.edge_index, feats, dataset.sink_node_mask, ift=True)
            probs = prob.expand_as(paths)
            loss = -probs[paths.bool()].log().sum()

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
