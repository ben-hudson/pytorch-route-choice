import networkx as nx
import pytest

from markov import load_rl_tutorial_network


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
