try:
    from feature_benchmark.cli import parse_args
    from feature_benchmark.runner import run_benchmark
except ImportError as exc:
    raise SystemExit(f"\nMissing import: {exc}")


if __name__ == "__main__":
    run_benchmark(parse_args())
