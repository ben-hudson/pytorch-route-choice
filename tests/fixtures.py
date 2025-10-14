import networkx as nx
import pytest


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
    G = nx.MultiDiGraph()
    G.add_node("o", pos=(0, 0))
    G.add_node("A", pos=(1, 0))
    G.add_node("B", pos=(2, 0))
    G.add_node("C", pos=(3, 0))
    G.add_node("D", pos=(4, 0))
    G.add_node("E", pos=(0, 1))
    G.add_node("F", pos=(1, 1))
    G.add_node("H", pos=(2, 1))
    G.add_node("I", pos=(3, 1))
    G.add_node("G", pos=(1, 2))
    G.add_node("d", pos=(4, 2))

    G.add_edge("o", "A", travel_time=0.3, flow=87.01)
    G.add_edge("A", "B", travel_time=0.1, flow=49.63)
    G.add_edge("B", "C", travel_time=0.1, flow=25.10)
    G.add_edge("C", "D", travel_time=0.3, flow=0.12)
    G.add_edge("o", "E", travel_time=0.4, flow=12.99)
    G.add_edge("A", "F", travel_time=0.1, flow=37.39)
    G.add_edge("B", "H", travel_time=0.2, flow=24.53)
    G.add_edge("C", "I", travel_time=0.1, flow=18.21)
    G.add_edge("C", "d", travel_time=0.9, flow=6.77)
    G.add_edge("D", "d", travel_time=2.6, flow=0.12)
    G.add_edge("E", "G", travel_time=0.3, flow=12.99)
    G.add_edge("F", "G", travel_time=0.3, flow=12.86)
    G.add_edge("F", "H", travel_time=0.2, flow=24.53)
    G.add_edge("H", "d", travel_time=0.5, flow=30.70)
    G.add_edge("H", "I", travel_time=0.2, flow=30.40)
    G.add_edge("I", "d", travel_time=0.3, flow=48.60)
    G.add_edge("G", "H", travel_time=0.6, flow=12.04)
    G.add_edge("G", "d", travel_time=0.7, flow=13.60)
    G.add_edge("G", "d", travel_time=2.8, flow=0.2)

    return G
