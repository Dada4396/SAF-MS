# Data Format

## Array Contract

SparseMSFlow consumes `.npy` files containing a three-dimensional NumPy array in
one of two layouts:

- `samples x 2 x 512` (channel first); or
- `samples x 512 x 2` (channel last).

The loader transposes channel-last arrays and returns a C-contiguous `int64`
array shaped `samples x 2 x 512`. Object arrays and pickle-backed data are not
accepted. Every other rank, channel count, or window length raises an error.

For each window, channel 0 stores aligned mass/position values and channel 1
stores aligned intensity values. Values are expected to be integer-quantized
before loading. The public API does not define a physical-unit scaling factor;
datasets must document their own reversible quantization convention.

## Development Split

`split_windows(windows, seed)` first applies a seeded permutation, then assigns
80% of samples to training, 10% to validation, and the remainder to the internal
test partition. Reusing the same array order and seed reproduces the split.

This development split is not a substitute for an independent test set.
Instrument, acquisition batch, or biological-source boundaries must be applied
before window-level splitting when leakage across those groups is possible.

## Excluded Data

This repository does not contain raw vendor files, converted research spectra,
precomputed windows, dataset manifests, or private filesystem locations. NumPy
arrays and compressed containers are ignored by Git by default.
