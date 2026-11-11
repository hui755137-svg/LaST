import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from last.data import (
    build_period_node_context,
    latent_target_statistics,
)
from last.trainer import train_period


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_logger(path, name):
    logger = logging.getLogger(f"last.{name}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    for handler in (logging.FileHandler(path / f"{name}.log"), logging.StreamHandler(sys.stdout)):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Reproduce one LaST main-result run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", default="main")
    parser.add_argument("--num-workers", type=int, default=None)
    cli = parser.parse_args()

    config_path = Path(cli.config).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    args = SimpleNamespace(**config)
    args.dataset = config_path.stem
    args.seed = cli.seed
    args.logname = cli.logname
    if cli.num_workers is not None:
        args.num_workers = cli.num_workers
    args.device = (
        torch.device(f"cuda:{cli.gpuid}")
        if torch.cuda.is_available() and cli.gpuid != -1
        else torch.device("cpu")
    )

    data_root = Path(cli.data_root).resolve() / args.dataset
    args.fast_data_path = data_root / "FastData"
    args.graph_path = data_root / "graph"
    args.model_path = Path(cli.output_root).resolve() / args.dataset
    args.path = args.model_path / f"{args.logname}-{args.seed}"
    args.path.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    args.logger = build_logger(args.path, args.logname)
    args.logger.info("params: %s", vars(args))
    args.result = {
        horizon: {metric: {} for metric in ("MAE", "RMSE", "MAPE")}
        for horizon in ("3", "6", "12", "Avg")
    }

    for year in range(args.begin_year, args.end_year + 1):
        with np.load(args.graph_path / f"{year}_adj.npz") as graph_file:
            adj = graph_file["x"]
        with np.load(args.fast_data_path / f"{year}.npz", allow_pickle=True) as inputs:
            args.year = year
            args.graph_size = adj.shape[0]
            args.adj = torch.from_numpy(
                adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
            ).float().to(args.device)
            context = build_period_node_context(inputs, args.adj.cpu().numpy())
            args.year_node_context = torch.from_numpy(context).float().to(args.device)
            args.ild_latent_y_mean, args.ild_latent_y_std = latent_target_statistics(inputs)
            args.logger.info("[*] Year %s data and 9-dimensional context loaded", year)
            train_period(inputs, args)

    summary = {}
    for horizon in ("3", "6", "12", "Avg"):
        summary[horizon] = {}
        for metric in ("MAE", "RMSE", "MAPE"):
            values = args.result[horizon][metric]
            summary[horizon][metric] = {
                "mean": float(np.mean(list(values.values()))),
                "periods": values,
            }
    total_time = sum(
        value["total_time"]
        for key, value in args.result.items()
        if isinstance(key, int)
    )
    args.logger.info("total time: %.4f", total_time)
    for horizon in ("3", "6", "12", "Avg"):
        for metric in ("MAE", "RMSE", "MAPE"):
            args.logger.info(
                "%s %s periods:%s mean:%.6f",
                horizon,
                metric,
                args.result[horizon][metric],
                summary[horizon][metric]["mean"],
            )

    metrics = {
        "dataset": args.dataset,
        "seed": args.seed,
        "begin_year": args.begin_year,
        "end_year": args.end_year,
        "config": vars(args).copy(),
        "summary": summary,
        "period_results": {
            str(year): value
            for year, value in args.result.items()
            if isinstance(year, int)
        },
        "total_time": total_time,
    }
    metrics["config"]["device"] = str(args.device)
    for key in ("fast_data_path", "graph_path", "model_path", "path"):
        metrics["config"][key] = str(metrics["config"][key])
    for key in ("adj", "year_node_context"):
        metrics["config"].pop(key, None)
    metrics["config"].pop("logger", None)
    with (args.path / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
        stream.write("\n")
    args.logger.info("Metrics saved to %s", args.path / "metrics.json")


if __name__ == "__main__":
    main()
