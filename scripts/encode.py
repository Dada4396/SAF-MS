"""Encode NumPy spectrum windows into a SparseMSFlow rANS container."""

import argparse

from sparse_ms_flow.data import load_spectrum_windows
from sparse_ms_flow.workflows import encode_array, load_checkpoint, save_container


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="input .npy windows")
    parser.add_argument("--output", required=True, help="output container")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    try:
        model = load_checkpoint(args.checkpoint, args.device)
        windows = load_spectrum_windows(args.input)
        payloads = encode_array(model, windows, args.precision)
        save_container(args.output, payloads, args.precision)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"latents={len(payloads)} container={args.output}")


if __name__ == "__main__":
    main()
