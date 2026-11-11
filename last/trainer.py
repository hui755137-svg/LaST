import math
import os
import os.path as osp
from contextlib import contextmanager, nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from .data import SpatioTemporalDataset, masked_mae_np, record_metrics
from .model import LaST


class TrainableParameterEMA:
    def __init__(self, model, decay, parameter_names):
        self.decay = float(decay)
        self.parameter_names = tuple(parameter_names)
        parameters = dict(model.named_parameters())
        self.shadow = {
            name: parameters[name].detach().clone() for name in self.parameter_names
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        parameters = dict(model.named_parameters())
        for name in self.parameter_names:
            self.shadow[name].mul_(self.decay).add_(
                parameters[name].detach(), alpha=1.0 - self.decay
            )

    @contextmanager
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        self._backup = {
            name: parameters[name].detach().clone() for name in self.parameter_names
        }
        with torch.no_grad():
            for name in self.parameter_names:
                parameters[name].copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name in self.parameter_names:
                    parameters[name].copy_(self._backup[name])
            self._backup = None


def build_ema(model, args):
    if args.year == args.begin_year:
        names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (name.startswith("encoder.") or name.startswith("predictor."))
        )
    else:
        names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("adaptive_residual.")
        )
    if not names:
        raise ValueError("EMA selected no trainable parameters.")
    return TrainableParameterEMA(model, 0.9, names)


def load_previous_model(args):
    checkpoint_dir = osp.join(
        args.model_path, f"{args.logname}-{args.seed}", str(args.year - 1)
    )
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        stem, extension = osp.splitext(filename)
        if extension == ".pkl":
            checkpoints.append((float(stem), filename))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    checkpoint_name = min(checkpoints)[1]
    checkpoint_path = osp.join(checkpoint_dir, checkpoint_name)
    args.logger.info("[*] load from %s", checkpoint_path)
    model = LaST(args)
    state_dict = torch.load(checkpoint_path, map_location=args.device)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model.to(args.device)


def validate(model, args, loader, use_ema=None):
    context = use_ema.average_parameters(model) if use_ema is not None else nullcontext()
    with context:
        validation_loss = 0.0
        count = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.adj)
                validation_loss += float(
                    masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                )
                count += 1
        if count == 0:
            raise ValueError("Validation loader produced no batches.")
        return validation_loss / count


def fit_adaptive_handoff(model, args, train_loader, val_loader, path):
    model.prepare_first_year_adaptive_handoff()
    stage_path = osp.join(path, "adaptive_handoff_stage")
    os.makedirs(stage_path, exist_ok=True)
    optimizer = optim.AdamW(model.optimizer_param_groups(args))
    lowest_validation_loss = float("inf")
    best_model_path = None
    counter = 0
    total_time = 0.0
    use_time = []
    last_epoch = -1

    for epoch in range(args.adaptive_handoff_epoch):
        start_time = datetime.now()
        model.train()
        training_loss = 0.0
        batch_count = 0
        loss_details = {}
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred, aux_loss, loss_details = model.forward_with_aux_loss(data, args.adj)
            forecast_loss = F.mse_loss(data.y, pred)
            loss = forecast_loss + aux_loss
            training_loss += float(loss.detach())
            batch_count += 1
            loss.backward()
            optimizer.step()
        elapsed = (datetime.now() - start_time).total_seconds()
        total_time += elapsed
        use_time.append(elapsed)
        training_loss /= batch_count
        validation_loss = validate(model, args, val_loader)
        args.logger.info(
            "first-year adaptive residual handoff epoch:%s training loss:%.4f validation loss:%.6f",
            epoch,
            training_loss,
            validation_loss,
        )
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = validation_loss
            best_model_path = osp.join(stage_path, f"{validation_loss:.6f}.pkl")
            torch.save({"model_state_dict": model.state_dict()}, best_model_path)
        else:
            counter += 1
            if counter > 5:
                last_epoch = epoch
                break
        last_epoch = epoch

    if best_model_path is None:
        raise ValueError("Adaptive handoff did not produce a checkpoint.")
    return best_model_path, total_time, use_time, last_epoch + 1


def test_model(model, args, test_loader):
    model.eval()
    predictions = []
    targets = []
    loss = 0.0
    count = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(args.device, non_blocking=True)
            pred = model(data, args.adj)
            loss += F.mse_loss(data.y, pred)
            pred, _ = to_dense_batch(pred, batch=data.batch)
            truth, _ = to_dense_batch(data.y, batch=data.batch)
            predictions.append(pred.cpu().data.numpy())
            targets.append(truth.cpu().data.numpy())
            count += 1
    args.logger.info("[*] loss:%.4f", loss / count)
    record_metrics(np.concatenate(targets, 0), np.concatenate(predictions, 0), args)


def train_period(inputs, args):
    path = osp.join(args.path, str(args.year))
    os.makedirs(path, exist_ok=True)
    loader_args = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(
        SpatioTemporalDataset(inputs, "train"), shuffle=True, **loader_args
    )
    val_loader = DataLoader(
        SpatioTemporalDataset(inputs, "val"), shuffle=False, **loader_args
    )
    test_loader = DataLoader(
        SpatioTemporalDataset(inputs, "test"), shuffle=False, **loader_args
    )
    args.logger.info("[*] Year %s Dataset load!", args.year)

    if args.year > args.begin_year:
        model = load_previous_model(args)
    else:
        model = LaST(args).to(args.device)
    model.prepare_continual_learning(args.year > args.begin_year)
    parameter_ema = build_ema(model, args)
    optimizer = optim.AdamW(model.optimizer_param_groups(args))

    args.logger.info("[*] Year %s Training start", args.year)
    lowest_validation_loss = 1e7
    counter = 0
    total_time = 0.0
    use_time = []
    model.train()
    for epoch in range(args.epoch):
        start_time = datetime.now()
        training_loss = 0.0
        batch_count = 0
        forecast_loss_value = math.nan
        aux_loss_dict = {}
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred, aux_loss, aux_loss_dict = model.forward_with_aux_loss(data, args.adj)
            forecast_loss = F.mse_loss(data.y, pred)
            forecast_loss_value = float(forecast_loss.detach())
            loss = forecast_loss + aux_loss
            training_loss += float(loss.detach())
            batch_count += 1
            loss.backward()
            optimizer.step()
            parameter_ema.update(model)

        elapsed = (datetime.now() - start_time).total_seconds()
        total_time += elapsed
        use_time.append(elapsed)
        training_loss /= batch_count
        validation_loss = validate(model, args, val_loader, parameter_ema)
        aux_info = " ".join(
            f"{name}:{float(value):.4f}" for name, value in aux_loss_dict.items()
        )
        args.logger.info(
            "epoch:%s, training loss:%.4f forecast loss:%.4f %s validation loss:%.4f",
            epoch,
            training_loss,
            forecast_loss_value,
            aux_info,
            validation_loss,
        )
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            checkpoint_path = osp.join(path, f"{lowest_validation_loss:.4f}.pkl")
            with parameter_ema.average_parameters(model):
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            counter += 1
            if counter > 5:
                break

    best_model_path = osp.join(path, f"{lowest_validation_loss:.4f}.pkl")
    model.load_state_dict(
        torch.load(best_model_path, map_location=args.device)["model_state_dict"]
    )
    model = model.to(args.device)

    handoff_time = 0.0
    handoff_use_time = []
    handoff_epochs = 0
    if args.year == args.begin_year:
        (
            handoff_best_path,
            handoff_time,
            handoff_use_time,
            handoff_epochs,
        ) = fit_adaptive_handoff(model, args, train_loader, val_loader, path)
        model.load_state_dict(
            torch.load(handoff_best_path, map_location=args.device)["model_state_dict"]
        )
        model = model.to(args.device)
        args.logger.info(
            "[*] First-year adaptive handoff best model:%s", handoff_best_path
        )

    test_model(model, args, test_loader)
    all_use_time = use_time + handoff_use_time
    args.result[args.year] = {
        "total_time": total_time + handoff_time,
        "average_time": sum(all_use_time) / len(all_use_time),
        "epoch_num": epoch + 1 + handoff_epochs,
    }
    args.logger.info(
        "Finished optimization, total time:%.2f s, best model:%s",
        total_time,
        best_model_path,
    )
