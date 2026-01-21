#!/usr/bin/env python3
"""
基于 experiment_cvae_yanjie.ipynb 的 CVAE 数据增强脚本
在指定的COAD数据上实现数据增强，输出到同一目录
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time, copy
from pathlib import Path
from typing import Union, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import joblib


# ========== Model Definition ==========
class CVAE(nn.Module): 
    """
    CVAE modeling p(markers | batch.id); VAE/maf sperate: p(markers) ; Cmaf

    Encoder input : [markers, batch_onehot]
    Decoder input : [z, batch_onehot]
    Output        : markers only (same dim as markers)
    """
    def __init__(self, num_markers, num_batches, latent_dim=32,
                 dropout=0.2, output_activation="identity"):
        super().__init__()

        self.num_markers = num_markers
        self.num_batches = num_batches
        self.latent_dim = latent_dim

        # ------------ Encoder ------------
        encoder_input_dim = num_markers + num_batches

        self.encoder_net = nn.Sequential(
            nn.Linear(encoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(True),
        )

        self.z_mean = nn.Linear(128, latent_dim)
        self.z_logvar = nn.Linear(128, latent_dim)

        # ------------ Decoder ------------
        decoder_input_dim = latent_dim + num_batches

        self.decoder_input = nn.Linear(decoder_input_dim, 128)
        self.decoder_net = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, num_markers),
        )

        # Output activation
        if output_activation == "softplus":
            self.output_activation = nn.Softplus()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

    # ----- helpers -----
    def encode(self, x_markers, batch_onehot):
        enc_in = torch.cat([x_markers, batch_onehot], dim=1)
        h = self.encoder_net(enc_in)
        mu = self.z_mean(h)
        logvar = self.z_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, batch_onehot):
        dec_in = torch.cat([z, batch_onehot], dim=1)
        h = self.decoder_input(dec_in)
        out = self.decoder_net(h)
        return self.output_activation(out)

    def forward(self, x_markers, batch_onehot, condition_dropout_rate=0.3):
        """
        x_markers:    [batch, num_markers], scaled
        batch_onehot: [batch, num_batches]
        """
        # encoder
        mu, logvar = self.encode(x_markers, batch_onehot)
        z = self.reparameterize(mu, logvar)

        # optional condition dropout (force z to carry more info)
        if self.training and (torch.rand(1).item() < condition_dropout_rate):
            batch_in = torch.zeros_like(batch_onehot)
        else:
            batch_in = batch_onehot

        # decoder
        x_recon = self.decode(z, batch_in)
        return z, mu, logvar, x_recon


# ========== Training Function ==========
def train_cvae(
    model,
    train_loader,
    num_epochs=500,
    learning_rate=1e-3,
    kl_weight=1.0,
    warmup_epochs=100,
    free_nats=0.0,
    grad_clip=1.0,
    early_stop=False,
    early_stop_patience=50,
    device="cpu",
    save_model_path=None,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_dict = {"train_recon": [], "train_kl": [], "train_total": [], "beta": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_recon = epoch_kl = epoch_total = 0.0

        # KL warmup
        beta = kl_weight * min(1.0, epoch / max(1, warmup_epochs))

        for x_markers, batch_onehot in train_loader:
            x_markers = x_markers.to(device)
            batch_onehot = batch_onehot.to(device)

            z, mu, logvar, x_recon = model(x_markers, batch_onehot)

            # reconstruction loss (markers are scaled)
            recon_loss = F.mse_loss(x_recon, x_markers, reduction="mean")

            # KL divergence per sample
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if free_nats and free_nats > 0:
                kl_summed = torch.clamp(kl_per_dim - free_nats, min=0.0).sum(dim=1).mean()
            else:
                kl_summed = kl_per_dim.sum(dim=1).mean()

            total_loss = recon_loss + beta * kl_summed

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_summed.item()
            epoch_total += total_loss.item()

        n_batches = len(train_loader)
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        avg_total = epoch_total / n_batches

        log_dict["train_recon"].append(avg_recon)
        log_dict["train_kl"].append(avg_kl)
        log_dict["train_total"].append(avg_total)
        log_dict["beta"].append(beta)

        print(
            f"Epoch {epoch+1}/{num_epochs} | β={beta:.3f} | "
            f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}"
        )

        # early stopping tracking
        if avg_total < best_loss:
            best_loss = avg_total
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif early_stop and (epoch - best_epoch >= early_stop_patience):
            print(f"Early stopping at epoch {epoch+1}, best at {best_epoch+1}")
            break

    elapsed = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed:.2f} min. Best epoch: {best_epoch+1}")

    model.load_state_dict(best_state)

    if save_model_path is not None:
        Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f"Saved best model to {save_model_path}")

    return log_dict, model


# ========== Data Preparation ==========
def prepare_data_for_cvae(df, batch_col="batch.id", log_transform=True):
    # separate markers vs batch
    # 排除 batch 列和 samples 列（样本ID）
    marker_cols = [c for c in df.columns if c not in [batch_col, "samples"]]
    num_markers = len(marker_cols)

    markers_raw = df[marker_cols].values.astype(float)

    # ---- optional log2(x+1) ----
    if log_transform:
        markers_for_model = np.log2(markers_raw + 1.0)
    else:
        markers_for_model = markers_raw

    # ---- Standardize markers ----
    scaler_markers = StandardScaler()
    X_markers = scaler_markers.fit_transform(markers_for_model)

    # ---- handle batch encoding ----
    unique_batches = df[batch_col].unique()
    num_unique = len(unique_batches)

    if num_unique == 2:
        # binary encoding (0 / 1)
        batch_mapping = {unique_batches[0]: 0, unique_batches[1]: 1}
        batch_encoded = df[batch_col].map(batch_mapping).values.reshape(-1, 1)

        batch_encoder = batch_mapping  # keep mapping for reference
        num_batches = 1
    else:
        # one-hot encoding
        batch_encoder = OneHotEncoder(sparse_output=False)
        batch_encoded = batch_encoder.fit_transform(df[[batch_col]])
        num_batches = batch_encoded.shape[1]

    # ---- tensors ----
    X_markers_tensor = torch.tensor(X_markers, dtype=torch.float32)
    batch_tensor = torch.tensor(batch_encoded.astype(float), dtype=torch.float32)

    return (
        marker_cols,
        num_markers,
        batch_encoder,
        num_batches,
        scaler_markers,
        X_markers_tensor,
        batch_tensor,
    )


def build_dataloader(X_markers_tensor, batch_tensor, batch_size=64):
    dataset = TensorDataset(X_markers_tensor, batch_tensor)
    # drop_last=True to keep BatchNorm happy
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


# ========== Generation Functions ==========
@torch.no_grad()
def generate_from_prior(
    model,
    scaler_markers,
    batch_encoder,
    batch_id,
    num_samples,
    marker_cols,
    temperature=1.0,
    device="cpu",
):
    model.eval()

    # ---- batch encoding ----
    if isinstance(batch_encoder, dict):
        # binary case
        batch_value = batch_encoder[batch_id]  # 0 or 1
        batch_encoded = np.full((num_samples, 1), batch_value)
    else:
        # one-hot case
        batch_onehot = batch_encoder.transform([[batch_id]])  # (1, num_batches)
        batch_encoded = np.repeat(batch_onehot, num_samples, axis=0)

    batch_tensor = torch.tensor(batch_encoded, dtype=torch.float32, device=device)

    # ---- sample from prior ----
    z = torch.randn(num_samples, model.latent_dim, device=device) * temperature

    # ---- decode ----
    with torch.no_grad():
        x_gen = model.decode(z, batch_tensor).cpu().numpy()

    # ---- inverse standardization ----
    x_gen = scaler_markers.inverse_transform(x_gen)
    
    # ---- build dataframe ----
    # marker_cols 已经排除了 batch 列，所以直接使用
    df_gen = pd.DataFrame(x_gen, columns=marker_cols)

    return df_gen


# ========== Main Augmentation Function ==========
def augment_data_with_cvae(
    data: Union[str, pd.DataFrame],
    output_path: str,
    output_filename: str,
    batch_col: str = "groups",
    samples_per_batch: int = 500,
    kl_weight: float = 0.05,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3,
    warmup_epochs: int = 50,
    batch_size: int = 64,
    early_stop: bool = True,
    early_stop_patience: int = 200,
    latent_dim: int = 32,
    dropout: float = 0.2,
    log_transform: bool = True,
    device: str = "cpu",
    save_model_path: Optional[str] = None,
    save_training_log: bool = True,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    CVAE数据增强主函数
    
    参数:
        data: 输入数据，可以是文件路径（str）或DataFrame
        output_path: 输出路径
        output_filename: 输出文件名（不含扩展名）
        batch_col: batch列名，默认为"groups"
        samples_per_batch: 每个batch生成的样本数，默认为500
        kl_weight: KL散度权重，默认为0.05
        num_epochs: 训练轮数，默认为1000
        learning_rate: 学习率，默认为1e-3
        warmup_epochs: KL warmup轮数，默认为50
        batch_size: 训练batch size，默认为64
        early_stop: 是否使用early stopping，默认为True
        early_stop_patience: early stopping patience，默认为200
        latent_dim: 潜在空间维度，默认为32
        dropout: dropout率，默认为0.2
        log_transform: 是否进行log变换，默认为True
        device: 设备（"cpu"或"cuda"），默认为"cpu"
        save_model_path: 模型保存路径，如果为None则不保存，默认为None
        save_training_log: 是否保存训练日志，默认为True
        verbose: 是否打印详细信息，默认为True
        **kwargs: 其他参数（如传递给pd.read_csv的参数）
    
    返回:
        pd.DataFrame: 增强后的数据
    """
    if verbose:
        print("=" * 60)
        print("CVAE Data Augmentation")
        print("=" * 60)
        print(f"Output path: {output_path}")
        print(f"Output filename: {output_filename}")
        print(f"KL weight: {kl_weight}")
        print(f"Epochs: {num_epochs}")
        print(f"Samples per batch: {samples_per_batch}")
        print("=" * 60)
    
    # ====== Load data ======
    if verbose:
        print("\n1. Loading data...")
    if isinstance(data, str):
        df = pd.read_csv(data, **kwargs)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data参数必须是文件路径（str）或DataFrame")
    
    if verbose:
        print(f"   Loaded {len(df)} samples with {len(df.columns)-2} markers")
        print(f"   Batch column: {batch_col}")
        print(f"   Unique batches: {sorted(df[batch_col].unique())}")
    
    # ====== Prepare data ======
    if verbose:
        print("\n2. Preparing data...")
    (
        marker_cols,
        num_markers,
        batch_encoder,
        num_batches,
        scaler_markers,
        X_markers_tensor,
        batch_tensor,
    ) = prepare_data_for_cvae(df, batch_col=batch_col, log_transform=log_transform)
    
    if verbose:
        print(f"   Number of markers: {num_markers}")
        print(f"   Number of batch dimensions: {num_batches}")
    
    # ====== Build dataloader ======
    if verbose:
        print("\n3. Building dataloader...")
    train_loader = build_dataloader(X_markers_tensor, batch_tensor, batch_size=batch_size)
    if verbose:
        print(f"   Batch size: {batch_size}")
        print(f"   Number of batches per epoch: {len(train_loader)}")
    
    # ====== Instantiate model ======
    if verbose:
        print("\n4. Creating CVAE model...")
    model = CVAE(
        num_markers=num_markers,
        num_batches=num_batches,
        latent_dim=latent_dim,
        dropout=dropout,
        output_activation="identity",   # markers are standardized
    )
    if verbose:
        print(f"   Model created with latent dimension: {latent_dim}")
    
    # ====== Train model ======
    if verbose:
        print("\n5. Training model...")
    
    log_dict, best_model = train_cvae(
        model=model,
        train_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        kl_weight=kl_weight,
        warmup_epochs=warmup_epochs,
        free_nats=0.0,
        grad_clip=1.0,
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        device=device,
        save_model_path=save_model_path,
    )
    
    # ====== Generate synthetic data ======
    if verbose:
        print("\n6. Generating synthetic data...")
    batch_ids = sorted(df[batch_col].unique())
    
    prior_dfs = []
    for bid in batch_ids:
        if verbose:
            print(f"   Generating {samples_per_batch} samples for batch {bid}...")
        df_prior_b = generate_from_prior(
            model=best_model,
            scaler_markers=scaler_markers,
            batch_encoder=batch_encoder,
            batch_id=bid,
            num_samples=samples_per_batch,
            marker_cols=marker_cols,
            temperature=1.0,
            device=device,
        )
        # 添加 batch 列
        df_prior_b[batch_col] = bid
        prior_dfs.append(df_prior_b)
    
    # Concatenate all batches
    df_prior_all = pd.concat(prior_dfs, ignore_index=True)
    
    # ====== Transform back to raw scale ======
    if verbose:
        print("\n7. Transforming to raw scale...")
    
    if log_transform:
        # Inverse log transform
        markers_raw = np.maximum(2**df_prior_all[marker_cols] - 1, 0)   # avoid tiny negatives
        df_prior_raw = df_prior_all.copy()
        df_prior_raw[marker_cols] = np.round(markers_raw)
    else:
        df_prior_raw = df_prior_all.copy()
    
    if verbose:
        print(f"   Generated {len(df_prior_raw)} samples total")
        print(f"   Batch distribution:")
        print(df_prior_raw[batch_col].value_counts().sort_index())
    
    # ====== Save results ======
    if verbose:
        print("\n8. Saving results...")
    
    # 确保输出路径存在
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_filepath = output_path / f"{output_filename}.csv"
    df_prior_raw.to_csv(output_filepath, index=False)
    if verbose:
        print(f"   Saved to: {output_filepath}")
    
    # Save training log
    if save_training_log:
        log_df = pd.DataFrame(log_dict)
        log_filepath = output_path / f"{output_filename}_loss.csv"
        log_df.to_csv(log_filepath, index=False)
        if verbose:
            print(f"   Training log saved to: {log_filepath}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Data augmentation completed successfully!")
        print("=" * 60)
    
    return df_prior_raw
