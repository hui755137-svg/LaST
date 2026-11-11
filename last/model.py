import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchGCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        self.weight_self = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        self.weight_self.reset_parameters()

    def forward(self, x, adj):
        if x.device != adj.device:
            adj = adj.to(x.device)
        return self.weight_neigh(torch.matmul(adj, x)) + self.weight_self(x)


class GraphObservationEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout):
        super().__init__()
        self.temporal_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.graph_conv = BatchGCNConv(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = self.temporal_proj(x)
        h = self.graph_conv(h, adj)
        h = self.norm(F.gelu(h))
        return self.dropout(h)


class GraphObservationDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        self.graph_conv = BatchGCNConv(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, adj):
        h = self.graph_conv(z, adj)
        h = self.norm(F.gelu(h))
        return self.head(h)


class LatentSTGNNPredictor(nn.Module):
    def __init__(self, latent_dim, dropout, kernel_size, dilation):
        super().__init__()
        self.gcn1 = BatchGCNConv(latent_dim, latent_dim)
        self.tcn = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.gcn2 = BatchGCNConv(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, adj):
        h = F.gelu(self.gcn1(z, adj))
        batch, nodes, channels = h.shape
        h = self.tcn(h.reshape(batch * nodes, 1, channels)).reshape(batch, nodes, channels)
        h = self.gcn2(F.gelu(h), adj)
        return self.dropout(self.norm(h + z))


class GraphLocalPropagationPredictor(nn.Module):
    def __init__(self, latent_dim, dropout, kernel_size, dilation):
        super().__init__()
        self.gcn1 = BatchGCNConv(latent_dim, latent_dim)
        self.tcn = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.gcn2 = BatchGCNConv(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, adj):
        h = F.gelu(self.gcn1(z, adj))
        batch, nodes, channels = h.shape
        h = self.tcn(h.reshape(batch * nodes, 1, channels)).reshape(batch, nodes, channels)
        h = self.gcn2(F.gelu(h), adj)
        return self.dropout(self.norm(h + z))


class SurfaceObservationCalibrator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, rank):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.graph_down = nn.Linear(hidden_dim, rank, bias=False)
        self.graph_up = nn.Linear(rank, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim + input_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, adj):
        h = F.gelu(self.input(features))
        propagated = torch.matmul(adj.unsqueeze(0), h)
        h = F.gelu(h + self.graph_up(self.graph_down(propagated)))
        h = self.out_norm(h)
        return self.output(torch.cat([h, features], dim=-1))


class LaST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.latent_loss_weight = args.latent_loss_weight
        self.stable_forecast_loss_weight = args.stable_forecast_loss_weight
        self.adaptive_residual_energy_weight = args.adaptive_residual_energy_weight
        self.adaptive_residual_lr = args.adaptive_residual_lr
        self._adaptive_handoff_active = False

        # These three constructors preserve the final experiment's RNG stream.
        # Its inherited prototypes were replaced by the final modules below.
        GraphObservationEncoder(
            args.gcn["in_channel"], args.latent_dim, args.encoder_hidden_dim, args.dropout
        )
        LatentSTGNNPredictor(
            args.latent_dim, args.dropout, args.tcn["kernel_size"], args.tcn["dilation"]
        )
        GraphObservationDecoder(
            args.latent_dim, args.y_len, args.decoder_hidden_dim, args.dropout
        )

        self.encoder = GraphObservationEncoder(
            args.gcn["in_channel"], args.latent_dim, args.encoder_hidden_dim, args.dropout
        )
        self.predictor = GraphLocalPropagationPredictor(
            args.latent_dim, args.dropout, args.tcn["kernel_size"], args.tcn["dilation"]
        )
        self.decoder = GraphObservationDecoder(
            args.latent_dim, args.y_len, args.decoder_hidden_dim, args.dropout
        )

        self.stable_anchor_encoder = copy.deepcopy(self.encoder)
        self.stable_anchor_predictor = copy.deepcopy(self.predictor)
        self.stable_anchor_decoder = copy.deepcopy(self.decoder)
        for module in (
            self.stable_anchor_encoder,
            self.stable_anchor_predictor,
            self.stable_anchor_decoder,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.adaptive_residual = SurfaceObservationCalibrator(
            input_dim=args.gcn["in_channel"] + args.year_node_context_dim,
            output_dim=args.y_len,
            hidden_dim=args.adaptive_residual_hidden_dim,
            rank=args.graph_prop_rank,
        )

        first_year_mean = float(args.ild_latent_y_mean)
        first_year_std = float(args.ild_latent_y_std)
        if not math.isfinite(first_year_mean) or not math.isfinite(first_year_std) or first_year_std <= 0:
            raise ValueError("Latent-target normalization requires finite mean and positive std.")
        self.register_buffer("stable_latent_target_mean", torch.tensor(first_year_mean))
        self.register_buffer("stable_latent_target_std", torch.tensor(first_year_std))

    def _reshape_x(self, data, adj):
        return data.x.reshape(-1, adj.shape[0], self.args.gcn["in_channel"])

    def _reshape_y(self, data, adj):
        return data.y.reshape(-1, adj.shape[0], self.args.y_len)

    @staticmethod
    def _flatten_nodes(x):
        return x.reshape(-1, x.shape[-1])

    def _latent_target_y(self, y):
        mean = self.stable_latent_target_mean.to(device=y.device, dtype=y.dtype)
        std = self.stable_latent_target_std.to(device=y.device, dtype=y.dtype)
        return (y - mean) / std

    def _adaptive_is_active(self):
        return self._adaptive_handoff_active or self.args.year > self.args.begin_year

    def _stable_forecast(self, z_x, adj):
        return self._flatten_nodes(self.decoder(self.predictor(z_x, adj), adj))

    def _adaptive_forecast_residual(self, x, adj):
        if not self._adaptive_is_active():
            return None
        context = self.args.year_node_context.to(device=adj.device, dtype=adj.dtype)
        if context.shape != (adj.shape[0], self.args.year_node_context_dim):
            raise ValueError("Period-level node context does not match the active graph.")
        batch = x.shape[0]
        context = context.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat([x.detach(), context], dim=-1)
        return self._flatten_nodes(self.adaptive_residual(features, adj))

    def prepare_continual_learning(self, is_continual_year):
        for module in (self.encoder, self.predictor, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = not is_continual_year
        for parameter in self.adaptive_residual.parameters():
            parameter.requires_grad = is_continual_year

    @torch.no_grad()
    def _sync_stable_anchor(self):
        self.stable_anchor_encoder.load_state_dict(self.encoder.state_dict())
        self.stable_anchor_predictor.load_state_dict(self.predictor.state_dict())
        self.stable_anchor_decoder.load_state_dict(self.decoder.state_dict())

    def prepare_first_year_adaptive_handoff(self):
        self._sync_stable_anchor()
        for module in (self.encoder, self.predictor, self.decoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
        for parameter in self.adaptive_residual.parameters():
            parameter.requires_grad = True
        self._adaptive_handoff_active = True

    def optimizer_param_groups(self, args):
        groups = []
        for module, lr in (
            (self.encoder, args.encoder_lr),
            (self.decoder, args.decoder_lr),
            (self.predictor, args.predictor_lr),
            (self.adaptive_residual, self.adaptive_residual_lr),
        ):
            parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
            if parameters:
                groups.append({"params": parameters, "lr": lr})
        return groups

    def forward(self, data, adj):
        x = self._reshape_x(data, adj)
        z_x = self.encoder(x, adj)
        y_stable = self._stable_forecast(z_x, adj)
        y_adaptive = self._adaptive_forecast_residual(x, adj)
        return y_stable if y_adaptive is None else y_stable + y_adaptive

    def forward_with_aux_loss(self, data, adj):
        x = self._reshape_x(data, adj)
        y = self._reshape_y(data, adj)
        z_x = self.encoder(x, adj)
        z_y = self.encoder(self._latent_target_y(y), adj)
        z_y_hat = self.predictor(z_x, adj)
        y_stable = self._flatten_nodes(self.decoder(z_y_hat, adj))
        y_adaptive = self._adaptive_forecast_residual(x, adj)

        stable_for_output = y_stable.detach() if y_adaptive is not None else y_stable
        pred = stable_for_output if y_adaptive is None else stable_for_output + y_adaptive
        latent_loss = F.mse_loss(z_y_hat, z_y.detach())
        stable_loss = F.mse_loss(y_stable, self._flatten_nodes(y))
        aux_loss = (
            self.latent_loss_weight * latent_loss
            + self.stable_forecast_loss_weight * stable_loss
        )
        loss_dict = {
            "latent_loss": latent_loss.detach(),
            "stable_forecast_loss": stable_loss.detach(),
        }
        if y_adaptive is not None:
            residual_target = self._flatten_nodes(y) - y_stable.detach()
            equivalent_residual_mse = F.mse_loss(y_adaptive, residual_target)
            scale = float(self.args.ild_latent_y_std)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Adaptive residual normalization requires a positive target std.")
            adaptive_energy = (y_adaptive / scale).square().mean()
            aux_loss = aux_loss + self.adaptive_residual_energy_weight * adaptive_energy
            loss_dict.update(
                {
                    "equivalent_residual_mse": equivalent_residual_mse.detach(),
                    "adaptive_residual_energy": adaptive_energy.detach(),
                }
            )
        return pred, aux_loss, loss_dict
