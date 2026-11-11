import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_job(job):
    dataset = job["dataset"]
    seed = job["seed"]
    gpu = job["gpu"]
    output_dir = job["output_root"] / dataset / f"main-{seed}"
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"SKIP {dataset} seed{seed} gpu{gpu}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(ROOT / "configs" / f"{dataset}.json"),
        "--data-root",
        str(job["data_root"]),
        "--output-root",
        str(job["output_root"]),
        "--seed",
        str(seed),
        "--gpuid",
        str(gpu),
        "--num-workers",
        str(job["num_workers"]),
    ]
    with (output_dir / "runner.log").open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"{dataset} seed{seed} failed; see {output_dir / 'runner.log'}")
    print(f"DONE {dataset} seed{seed} gpu{gpu}", flush=True)


def aggregate(output_root, datasets, seeds):
    rows = []
    for dataset in datasets:
        runs = []
        for seed in seeds:
            path = output_root / dataset / f"main-{seed}" / "metrics.json"
            with path.open(encoding="utf-8") as stream:
                runs.append(json.load(stream))
        for horizon in ("3", "6", "12", "Avg"):
            for metric in ("MAE", "RMSE", "MAPE"):
                values = [run["summary"][horizon][metric]["mean"] for run in runs]
                rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "metric": metric,
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values),
                        "n": len(values),
                        "values": values,
                    }
                )
    output_path = output_root / "main_results.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    for row in rows:
        if row["horizon"] == "Avg":
            print(
                f"{row['dataset']:4s} Avg {row['metric']:4s} "
                f"{row['mean']:.3f} ± {row['std']:.3f}"
            )
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run and aggregate the 15 LaST main-result jobs."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--datasets", default="AIR,CA,PEMS")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not datasets or set(datasets) - {"AIR", "CA", "PEMS"}:
        raise SystemExit("datasets must be a nonempty subset of AIR,CA,PEMS")
    if not seeds or not gpus:
        raise SystemExit("seeds and gpus must be nonempty")
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.aggregate_only:
        aggregate(args.output_root, datasets, seeds)
        return

    jobs = [
        {
            "dataset": dataset,
            "seed": seed,
            "data_root": args.data_root,
            "output_root": args.output_root,
            "num_workers": args.num_workers,
        }
        for dataset in datasets
        for seed in seeds
    ]
    print(
        f"jobs={len(jobs)} datasets={datasets} seeds={seeds} gpus={gpus} "
        f"data_root={args.data_root} output_root={args.output_root}",
        flush=True,
    )
    if args.dry_run:
        return

    work = queue.Queue()
    for job in jobs:
        work.put(job)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            try:
                run_job({**job, "gpu": gpu})
            except Exception as error:
                with lock:
                    failures.append(error)
                print(f"FAIL {error}", flush=True)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")
    aggregate(args.output_root, datasets, seeds)


if __name__ == "__main__":
    main()
