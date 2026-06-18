# Reproducibility

## Included Verification

Install the development dependencies and run:

```bash
python -m pytest -q
python examples/smoke_test.py
```

The tests cover package imports, attention shapes and gradients, conditioner
gradients, exact integer-flow inversion, entropy gradients, rANS round trips,
data validation, configuration loading, and command availability. The smoke
example runs a synthetic forward/backward pass and exact inverse flow.

## Seeds And Configuration

`configs/default.yaml` records the default architecture and training settings.
The default seed is 17. Dataset splitting uses a local NumPy random generator;
synthetic batches use a local PyTorch generator. Commands accept explicit paths
and do not select a GPU automatically.

Exact training trajectories can still vary across hardware, PyTorch versions,
and accelerated kernels. Record the environment, resolved YAML, command line,
checkpoint hash, data preparation, and split grouping for an experiment.

## Checkpoints And Data

Running `scripts/train.py` creates a checkpoint containing the model
configuration and state dictionary. Encoding and decoding use that compatible
checkpoint.

No raw or processed research dataset is distributed. The public data boundary
is the validated integer NumPy array described in `docs/data-format.md`. Keep an
independent test set separate from the seeded 80/10/10 development split.
