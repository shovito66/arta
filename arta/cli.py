"""Command-line entry point for Artifact-Robust Temporal Alignment."""

from __future__ import annotations

import logging

from .arta import build_arg_parser, run_pipeline


def main() -> None:
    """Run the ARTA full-record pipeline from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    run_pipeline(args)


if __name__ == "__main__":
    main()

