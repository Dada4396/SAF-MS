"""Decode a SAF-MS rANS container into NumPy windows."""

import argparse

import numpy as np

from saf_ms.workflows import decode_array, load_checkpoint, load_container


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="input container")
    parser.add_argument("--output", required=True, help="output .npy windows")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    try:
        model = load_checkpoint(args.checkpoint, args.device)
        precision, payloads = load_container(args.input)
        windows = decode_array(model, payloads, precision)
        np.save(args.output, windows, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"windows={len(windows)} output={args.output}")


if __name__ == "__main__":
    main()
