from __future__ import annotations
import argparse
from school_credit.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home Credit School Edition 10K")
    parser.add_argument("--data-dir", required=True, help="Ruta a csv_files o csv_files/train")
    parser.add_argument("--profile", choices=["smoke", "school", "ensemble"], default="school")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=250)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        profile_name=args.profile,
        output_dir=args.output_dir,
        random_state=args.random_state,
        max_features=args.max_features,
        threads=args.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
