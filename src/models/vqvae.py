"""VQ-VAE tokenizer (standalone component).

Compresses 128×128×3 frames into a 16×16 grid of discrete codebook indices.
Not used in the dynamics pipeline — trained separately for analysis, evaluation,
and as a portfolio piece demonstrating understanding of discrete representations.

Architecture (from docs/build_spec.md §2.2):
  Encoder: 3 strided convs (3→64→128→256) + 2 ResBlocks + 1×1 conv + L2 norm
  Quantizer: 512 codebook entries, 256-dim, cosine similarity, EMA updates
  Decoder: mirror of encoder with ConvTranspose2d, Sigmoid output

Reference: docs/build_spec.md §2.2, docs/foundations_guide.md Parts 2-4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEResBlock(nn.Module):
    """Simple residual block for VQ-VAE encoder/decoder (no conditioning).

    Unlike the U-Net's ResBlock, this has no AdaGN — the VQ-VAE is a pure
    autoencoder with no time or action conditioning.

    Architecture: ReLU → Conv3×3 → ReLU → Conv3×3 + skip

    Args:
        channels: Number of input/output channels (must be equal).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map, shape [B, C, H, W].

        Returns:
            Output feature map, shape [B, C, H, W].
        """
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return h + x


class Encoder(nn.Module):
    """VQ-VAE encoder: compress 3×H×W image to 256×(H/8)×(W/8) features.

    Architecture:
        Conv2d(3→64, k=4, s=2, p=1) + ReLU   → [B, 64, H/2, W/2]
        Conv2d(64→128, k=4, s=2, p=1) + ReLU  → [B, 128, H/4, W/4]
        Conv2d(128→256, k=4, s=2, p=1) + ReLU → [B, 256, H/8, W/8]
        ResBlock(256) × 2                       → [B, 256, H/8, W/8]
        Conv2d(256→256, k=1, s=1) + L2Norm     → [B, 256, H/8, W/8]

    The final L2 normalization projects encoder outputs onto the unit sphere,
    matching the L2-normalized codebook. This prevents the encoder from
    "escaping" the codebook by increasing output magnitude — a key fix for
    codebook collapse (docs/foundations_guide.md Part 4).

    Args:
        in_channels: Input image channels (3 for RGB).
        hidden_channels: Channel progression for conv stack.
        embed_dim: Output embedding dimension (must match codebook dim).
        num_res_blocks: Number of residual blocks after final conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: list[int] = [64, 128, 256],
        embed_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # Strided conv stack: downsample 3 times (8× total)
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out

        # Residual blocks at bottleneck resolution
        for _ in range(num_res_blocks):
            layers.append(VQVAEResBlock(hidden_channels[-1]))

        # 1×1 conv to project to embed_dim (identity if already matching)
        layers.append(nn.Conv2d(hidden_channels[-1], embed_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to continuous features, L2-normalized.

        Args:
            x: Input image, shape [B, 3, H, W], values in [0, 1].

        Returns:
            L2-normalized features, shape [B, embed_dim, H/8, W/8].
        """
        z_e = self.net(x)  # [B, embed_dim, H/8, W/8]
        # L2 normalize along the channel dimension
        z_e = F.normalize(z_e, p=2, dim=1)
        return z_e


class Decoder(nn.Module):
    """VQ-VAE decoder: reconstruct 3×H×W image from 256×(H/8)×(W/8) codes.

    Mirror of the encoder using ConvTranspose2d for upsampling.

    Architecture:
        ResBlock(256) × 2                              → [B, 256, H/8, W/8]
        ConvTranspose2d(256→128, k=4, s=2, p=1) + ReLU → [B, 128, H/4, W/4]
        ConvTranspose2d(128→64, k=4, s=2, p=1) + ReLU  → [B, 64, H/2, W/2]
        ConvTranspose2d(64→3, k=4, s=2, p=1) + Sigmoid → [B, 3, H, W]

    Final Sigmoid ensures output is in [0, 1] to match normalized input range.

    Args:
        out_channels: Output image channels (3 for RGB).
        hidden_channels: Channel progression (reverse of encoder).
        embed_dim: Input embedding dimension (must match codebook dim).
        num_res_blocks: Number of residual blocks before upsampling.
    """

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: list[int] = [64, 128, 256],
        embed_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # Residual blocks at bottleneck resolution
        for _ in range(num_res_blocks):
            layers.append(VQVAEResBlock(embed_dim))

        # Transposed conv stack: upsample 3 times (8× total)
        # Reverse the hidden_channels order for the decoder
        reversed_channels = list(reversed(hidden_channels))
        ch_in = reversed_channels[0]  # 256
        for ch_out in reversed_channels[1:]:  # 128, 64
            layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out

        # Final transpose conv to output channels
        layers.append(nn.ConvTranspose2d(ch_in, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized features to image.

        Args:
            z_q: Quantized features, shape [B, embed_dim, H/8, W/8].

        Returns:
            Reconstructed image, shape [B, 3, H, W], values in [0, 1].
        """
        return self.net(z_q)


class VectorQuantizer(nn.Module):
    """Vector quantization with EMA codebook updates, L2 normalization, dead code reset.

    This is the core of the VQ-VAE. It maintains a codebook of K vectors on the
    unit sphere. Each spatial position in the encoder output is replaced by its
    nearest codebook entry (cosine similarity, which equals dot product when both
    are L2-normalized). Gradients pass through via the straight-through estimator.

    The codebook is updated via Exponential Moving Average (EMA) — more stable
    than gradient-based learning. Dead codes (entries used fewer than threshold
    times) are periodically reset to random encoder outputs + small noise.

    Reference: docs/foundations_guide.md Part 4 (VQ-VAE, codebook collapse fixes).

    Args:
        codebook_size: Number of codebook entries (K).
        embed_dim: Dimension of each entry (D).
        ema_decay: EMA decay factor (γ = 0.99).
        commitment_cost: Weight β for commitment loss.
        dead_code_threshold: Reset entries used fewer than this many times.
        dead_code_reset_every: Steps between dead code checks.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
        dead_code_threshold: int = 2,
        dead_code_reset_every: int = 1000,
    ) -> None:
        super().__init__()

        self.K = codebook_size
        self.D = embed_dim
        self.ema_decay = ema_decay
        self.commitment_cost = commitment_cost
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_reset_every = dead_code_reset_every

        # Codebook: K entries of D dimensions, initialized on unit sphere
        codebook = torch.randn(codebook_size, embed_dim)
        codebook = F.normalize(codebook, p=2, dim=1)
        self.register_buffer("codebook", codebook)  # [K, D]

        # EMA state: running count and running sum for each code
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", codebook.clone())  # [K, D]

        # Usage tracking for dead code reset
        self.register_buffer("usage_count", torch.zeros(codebook_size))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder output using nearest codebook entry.

        Args:
            z_e: Encoder output, shape [B, D, H, W]. Expected to be L2-normalized.

        Returns:
            z_q: Quantized features (straight-through), shape [B, D, H, W].
            indices: Codebook indices, shape [B, H, W].
            commitment_loss: β * ||z_e - sg(z_q)||², scalar.
        """
        B, D, H, W = z_e.shape

        # Reshape: [B, D, H, W] → [B*H*W, D] for codebook lookup
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # [N, D]

        # L2 normalize (should already be normalized, but ensure)
        z_e_flat = F.normalize(z_e_flat, p=2, dim=1)

        # L2 normalize codebook (should already be, but ensure after EMA updates)
        codebook = F.normalize(self.codebook, p=2, dim=1)  # [K, D]

        # Cosine similarity = dot product (both on unit sphere)
        # Higher similarity = closer → use argmax
        similarity = z_e_flat @ codebook.t()  # [N, K]
        indices = similarity.argmax(dim=1)  # [N]

        # Look up quantized vectors
        z_q_flat = codebook[indices]  # [N, D]

        # Reshape back to spatial: [B, H, W, D] → [B, D, H, W]
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        # EMA codebook update (only during training)
        if self.training:
            self._ema_update(z_e_flat, indices.reshape(-1))
            self.global_step += 1

        # Straight-through estimator: forward uses z_q, backward uses z_e
        # z_q = z_e + (z_q - z_e).detach()  →  gradient goes straight to z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, commitment_loss

    @torch.no_grad()
    def _ema_update(self, z_e_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via exponential moving average of assigned encoder outputs.

        Instead of gradient descent on the codebook, we maintain a running average
        of the encoder outputs assigned to each entry. This is more stable and is
        the standard approach (docs/foundations_guide.md Part 4).

        Args:
            z_e_flat: Flat encoder outputs, shape [N, D].
            indices: Codebook assignments, shape [N].
        """
        γ = self.ema_decay

        # Count assignments per code in this batch
        one_hot = F.one_hot(indices, self.K).float()  # [N, K]
        batch_count = one_hot.sum(dim=0)  # [K]
        batch_sum = one_hot.t() @ z_e_flat  # [K, D]

        # EMA update
        self.ema_count.mul_(γ).add_(batch_count, alpha=1 - γ)
        self.ema_weight.mul_(γ).add_(batch_sum, alpha=1 - γ)

        # Update codebook: weight / count (with Laplace smoothing to avoid div by 0)
        n = self.ema_count.sum()
        smoothed_count = (
            (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
        )
        self.codebook.copy_(self.ema_weight / smoothed_count.unsqueeze(1))

        # L2 normalize codebook after update
        self.codebook.copy_(F.normalize(self.codebook, p=2, dim=1))

        # Track usage for dead code detection
        self.usage_count += batch_count

    @torch.no_grad()
    def maybe_reset_dead_codes(self, z_e: torch.Tensor) -> int:
        """Reset dead codebook entries if it's time to check.

        Call this every training step. Internally checks if dead_code_reset_every
        steps have passed since the last reset.

        Dead entries (used < threshold times since last check) are replaced with
        random encoder outputs + small noise, giving them a chance to become
        useful again. This prevents codebook collapse.

        Args:
            z_e: Current batch encoder outputs, shape [B, D, H, W].
                 Random samples from this are used to reinitialize dead codes.

        Returns:
            Number of codes reset (0 if not a reset step).
        """
        step = self.global_step.item()
        if step == 0 or step % self.dead_code_reset_every != 0:
            return 0

        # Identify dead codes
        dead_mask = self.usage_count < self.dead_code_threshold  # [K]
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            # Flatten encoder outputs for sampling
            z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.D)  # [N, D]

            # Sample random encoder outputs to replace dead codes
            N = z_e_flat.shape[0]
            sample_indices = torch.randint(0, N, (num_dead,), device=z_e.device)
            replacements = z_e_flat[sample_indices]

            # Add small noise for diversity
            replacements = replacements + 0.01 * torch.randn_like(replacements)
            replacements = F.normalize(replacements, p=2, dim=1)

            # Replace dead entries
            self.codebook[dead_mask] = replacements
            self.ema_weight[dead_mask] = replacements
            self.ema_count[dead_mask] = 1.0  # Give them a small initial count

        # Reset usage tracking for next window
        self.usage_count.zero_()

        return num_dead

    @torch.no_grad()
    def codebook_utilization(self) -> float:
        """Fraction of codebook entries that have been used since last reset.

        Returns:
            Utilization as a float in [0, 1]. Target is > 0.80.
        """
        used = (self.usage_count > 0).sum().item()
        return used / self.K


class VQVAE(nn.Module):
    """Complete VQ-VAE: Encoder → VectorQuantizer → Decoder.

    Standalone autoencoder that compresses frames into discrete codebook indices.
    Not part of the dynamics pipeline — used for evaluation and analysis.

    Args:
        in_channels: Input/output image channels (3 for RGB).
        hidden_channels: Channel progression in encoder/decoder.
        embed_dim: Codebook embedding dimension.
        codebook_size: Number of codebook entries.
        num_res_blocks: Residual blocks in encoder/decoder.
        ema_decay: EMA decay for codebook updates.
        commitment_cost: Weight for commitment loss.
        dead_code_threshold: Reset codes used fewer than this.
        dead_code_reset_every: Steps between dead code checks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: list[int] = [64, 128, 256],
        embed_dim: int = 256,
        codebook_size: int = 512,
        num_res_blocks: int = 2,
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
        dead_code_threshold: int = 2,
        dead_code_reset_every: int = 1000,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            num_res_blocks=num_res_blocks,
        )
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            ema_decay=ema_decay,
            commitment_cost=commitment_cost,
            dead_code_threshold=dead_code_threshold,
            dead_code_reset_every=dead_code_reset_every,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to quantized features and codebook indices.

        Args:
            x: Input image, shape [B, 3, H, W], values in [0, 1].

        Returns:
            z_q: Quantized features, shape [B, D, H/8, W/8].
            indices: Codebook indices, shape [B, H/8, W/8].
        """
        z_e = self.encoder(x)
        z_q, indices, _ = self.quantizer(z_e)
        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized features to image.

        Args:
            z_q: Quantized features, shape [B, D, H/8, W/8].

        Returns:
            Reconstructed image, shape [B, 3, H, W], values in [0, 1].
        """
        return self.decoder(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → quantize → decode.

        Args:
            x: Input image, shape [B, 3, H, W], values in [0, 1].

        Returns:
            x_hat: Reconstructed image, shape [B, 3, H, W].
            indices: Codebook indices, shape [B, H/8, W/8].
            recon_loss: MSE reconstruction loss, scalar.
            commitment_loss: Commitment loss (already weighted by β), scalar.
        """
        z_e = self.encoder(x)
        z_q, indices, commitment_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        return x_hat, indices, recon_loss, commitment_loss

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total VQ-VAE loss and return metrics.

        Total loss = reconstruction + commitment (codebook is EMA-updated, no gradient).

        Args:
            x: Input image, shape [B, 3, H, W], values in [0, 1].

        Returns:
            loss: Total loss, scalar.
            metrics: Dict with 'recon_loss', 'commitment_loss', 'total_loss',
                     'codebook_utilization'.
        """
        x_hat, indices, recon_loss, commitment_loss = self.forward(x)
        total_loss = recon_loss + commitment_loss

        # Dead code reset (safe to call every step — internally checks timing)
        z_e = self.encoder(x)  # Need encoder output for reset replacements
        self.quantizer.maybe_reset_dead_codes(z_e)

        metrics = {
            "recon_loss": recon_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "total_loss": total_loss.item(),
            "codebook_utilization": self.quantizer.codebook_utilization(),
        }

        return total_loss, metrics
