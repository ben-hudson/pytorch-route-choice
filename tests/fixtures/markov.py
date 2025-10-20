import networkx as nx
import pytest

from route_choice.markov.dataset import load_rl_tutorial_network


def load_small_acyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5803)
    G.add_node(2, value=-1.6867)
    G.add_node(3, value=-1.5)
    G.add_node(4, value=0.0)

    G.add_edge(1, 2, cost=1, prob=0.3308)
    G.add_edge(1, 4, cost=2, prob=0.6572)
    G.add_edge(1, 4, cost=6, prob=0.0120)
    G.add_edge(2, 3, cost=1.5, prob=0.2689)
    G.add_edge(2, 4, cost=2, prob=0.7311)
    G.add_edge(3, 4, cost=1.5, prob=1.0)

    return G


def load_small_cyclic_network():
    G = nx.MultiDiGraph()

    G.add_node(1, value=-1.5496)
    G.add_node(2, value=-1.5968)
    G.add_node(3, value=-1.1998)
    G.add_node(4, value=0.0)

    G.add_edge(1, 2, cost=1, prob=0.3509)
    G.add_edge(1, 4, cost=2, prob=0.6374)
    G.add_edge(1, 4, cost=6, prob=0.0117)
    G.add_edge(2, 3, cost=1.5, prob=0.3318)
    G.add_edge(2, 4, cost=2, prob=0.6682)
    G.add_edge(3, 4, cost=1.5, prob=0.7407)
    G.add_edge(3, 1, cost=1, prob=0.2593)

    return G


@pytest.fixture
def small_network(request: pytest.FixtureRequest):
    params = getattr(request, "param", {})
    if params.get("cyclic", False):
        G = load_small_cyclic_network()
    else:
        G = load_small_acyclic_network()

    return G


@pytest.fixture
def rl_tutorial_network():
    return load_rl_tutorial_network()


@pytest.fixture
def nrl_toy_network():
    G = nx.MultiDiGraph()

    G.add_node("a", scale=1.0)
    G.add_node("b", scale=0.5)  # this is the first nest
    G.add_node("c", scale=0.8)  # this is the second nest
    G.add_node("d", scale=1.0)

    G.add_edge("a", "b", cost=1, name="a")
    G.add_edge("b", "d", cost=1, name="a1")
    G.add_edge("b", "d", cost=2, name="a2")
    G.add_edge("b", "d", cost=3, name="a3")
    G.add_edge("a", "c", cost=2, name="b")
    G.add_edge("c", "d", cost=2, name="b1")
    G.add_edge("c", "d", cost=1.5, name="b2")
    G.add_edge("c", "d", cost=1, name="b3")

    return G
