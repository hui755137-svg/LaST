# LaST: Main Results Only

Minimal reproduction of the final frozen LaST main results on AIR, CA, and PEMS.

The fixed configuration is latent dimension 48, calibrator width 128, graph-propagation rank 32, and nine period-level node-context features. The encoder, latent predictor, and decoder are frozen after the first-period adaptive handoff; later periods update only the Surface Observation Calibrator.

## Data

Keep the existing cached data outside this repository:

```text
DATA_ROOT/
  AIR/{FastData,graph}/
  CA/{FastData,graph}/
  PEMS/{FastData,graph}/
```

Only `FastData/*.npz` and `graph/*_adj.npz` are read. Raw data and BenchmarkCache are not required.

## Run

```bash
pip install -r requirements.txt
python run_main.py --data-root /path/to/data --gpus 0
```

Use multiple GPUs to run one job per GPU:

```bash
python run_main.py --data-root /path/to/data --gpus 0,1,2
```

The complete run is 3 datasets × 5 seeds (42–46) = 15 jobs. Completed jobs are skipped on restart. Results are written to `results/`; the aggregate is `results/main_results.json`.

```bash
python run_main.py --dry-run
python run_main.py --aggregate-only
```

Expected five-seed average-horizon results (mean ± sample standard deviation):

| Dataset | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| AIR | 19.310 ± 0.157 | 30.658 ± 0.222 | 25.742 ± 0.166 |
| CA | 17.254 ± 0.050 | 26.880 ± 0.065 | 16.228 ± 0.045 |
| PEMS | 12.942 ± 0.016 | 20.786 ± 0.034 | 17.986 ± 0.105 |

To run one dataset or seed:

```bash
python run_main.py --datasets AIR --seeds 42 --gpus 0
```
