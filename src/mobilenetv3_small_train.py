from __future__ import annotations

from official_cnn_train import build_arg_parser, run_from_args


def main() -> None:
    parser = build_arg_parser(default_model_name="mobilenet_v3_small")
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
