import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class SpatioTemporalDataset(Dataset):
    def __init__(self, inputs, split):
        super().__init__()
        self.x = inputs[f"{split}_x"]
        self.y = inputs[f"{split}_y"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return Data(
            x=torch.Tensor(self.x[index].T),
            y=torch.Tensor(self.y[index].T),
        )


def build_period_node_context(inputs, adj):
    visible_series = np.asarray(inputs["train_x"], dtype=np.float32)
    if visible_series.ndim != 3 or visible_series.shape[2] != adj.shape[0]:
        raise ValueError("train_x must have shape [windows, sequence_len, num_nodes].")

    series = visible_series.reshape(-1, visible_series.shape[2])
    abs_series = np.abs(series)
    node_mean = series.mean(axis=0)
    node_std = series.std(axis=0)
    node_q25 = np.quantile(series, 0.25, axis=0)
    node_q75 = np.quantile(series, 0.75, axis=0)
    time_index = np.arange(series.shape[0], dtype=np.float32)
    centered_time = time_index - time_index.mean()
    time_scale = np.sum(centered_time * centered_time)
    if time_scale <= 0:
        raise ValueError("Not enough observations to compute the node trend.")
    trend = centered_time @ (series - node_mean[None, :]) / time_scale
    degree = (adj > 0).sum(axis=1).astype(np.float32)

    features = np.stack(
        [
            node_mean,
            node_std,
            node_q25,
            node_q75,
            trend,
            adj @ node_mean,
            adj @ node_std,
            np.log1p(degree),
            np.log1p(np.quantile(abs_series, 0.50, axis=0)),
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Period-level node context contains non-finite values.")
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    if np.any(std <= 0):
        raise ValueError("Period-level node context has a constant feature.")
    return ((features - mean) / std).astype(np.float32)


def latent_target_statistics(inputs):
    train_y = np.asarray(inputs["train_y"], dtype=np.float32)
    mean = np.float32(train_y.mean())
    std = np.float32(train_y.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Latent-target normalization requires finite train_y statistics.")
    return float(mean), float(std)


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype("float32")
    return np.not_equal(array, null_val).astype("float32")


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    return np.mean(np.nan_to_num(mask * np.abs(y_true - y_pred)))


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    return np.mean(np.nan_to_num(mask * (y_true - y_pred) ** 2))


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        return np.mean(np.nan_to_num(mask * mape)) * 100


def record_metrics(ground_truth, prediction, args):
    mae_list, rmse_list, mape_list = [], [], []
    args.logger.info("[*] year %s, testing", args.year)
    for horizon in range(1, 13):
        mae = masked_mae_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0)
        rmse = masked_mse_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :horizon], prediction[:, :, :horizon], 0)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        if horizon in (3, 6, 12):
            args.logger.info(
                "T:%d\tMAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f",
                horizon,
                mae,
                rmse,
                mape,
            )
            args.result[str(horizon)]["MAE"][args.year] = mae
            args.result[str(horizon)]["RMSE"][args.year] = rmse
            args.result[str(horizon)]["MAPE"][args.year] = mape
    args.result["Avg"]["MAE"][args.year] = float(np.mean(mae_list))
    args.result["Avg"]["RMSE"][args.year] = float(np.mean(rmse_list))
    args.result["Avg"]["MAPE"][args.year] = float(np.mean(mape_list))
    args.logger.info(
        "T:Avg\tMAE\t%.4f\tRMSE\t%.4f\tMAPE\t%.4f",
        np.mean(mae_list),
        np.mean(rmse_list),
        np.mean(mape_list),
    )
