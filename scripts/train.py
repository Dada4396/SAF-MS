"""Train a bounded SparseMSFlow model."""

import argparse
from dataclasses import replace

from sparse_ms_flow.config import load_config
from sparse_ms_flow.data import load_spectrum_windows
from sparse_ms_flow.model import SparseMSFlowModel
from sparse_ms_flow.workflows import save_checkpoint, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="NumPy spectrum windows")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-steps", type=int, help="bounded optimization steps")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", default="sparse_ms_flow.pt")
    args = parser.parse_args()

    try:
        model_config, settings = load_config(args.config)
        if args.device:
            settings = replace(settings, device=args.device)
        windows = None if args.synthetic else load_spectrum_windows(args.data)
        model = SparseMSFlowModel(model_config)
        losses = train_model(model, settings, windows, args.max_steps)
        save_checkpoint(model, args.output)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"steps={len(losses)} final_bits_per_value={losses[-1]:.6f}")
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()
