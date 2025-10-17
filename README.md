# pytorch-route-choice
This package has PyTorch implementations of some route choice models.
- `RecursiveLogitRouteChoice` (aliased to `MarkovRouteChoice`) implements "A link based network route choice model with unrestricted choice set" (Fosgerau et al., 2013).
    - It can also be used to implement "Maximum entropy inverse reinforcement learning" (Ziebart et al., 2008).
- `PerturbedUtilityRouteChoice` implements "A perturbed utility route choice model" (Fosgerau et al., 2022).

The implementations make use of sparse operations, allowing the models to scale to large networks.

## Installation
Run:
```
pip install git+https://github.com/ben-hudson/pytorch-route-choice
```

## Getting started
Check out the examples and tests for usage!
