"""Evaluate coding cost and flow invertibility."""

import argparse

import torch

from saf_ms.data import load_spectrum_windows
from saf_ms.workflows import evaluate_model, load_checkpoint, synthetic_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="NumPy spectrum windows")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    try:
        model = load_checkpoint(args.checkpoint, args.device)
        inputs = (
            synthetic_inputs(model.config)
            if args.synthetic
            else torch.from_numpy(load_spectrum_windows(args.data)).float()
        )
        metrics = evaluate_model(model, inputs)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"bits_per_value={metrics['bits_per_value']:.6f}")
    print(f"exact_round_trip={metrics['exact_round_trip']}")


if __name__ == "__main__":
    main()
