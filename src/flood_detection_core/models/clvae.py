"""
CLVAE Model Architecture Implementation

This script implements the CLVAE (Contrastive ConvLSTM Variational AutoEncoder)
model architecture as described in the model plan for unsupervised flood detection
on SAR time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for temporal sequence processing."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Combined convolution for all gates
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ConvLSTM cell."""
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split gates
        i_gate, f_gate, o_gate, g_gate = torch.split(gates, self.hidden_channels, dim=1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        # Update cell and hidden states
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new


class ConvLSTM(nn.Module):
    """ConvLSTM layer for processing temporal sequences."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.convlstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ConvLSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, time, channels, height, width)

        Returns
        -------
            Output tensor of shape (batch, time, hidden_channels, height, width)
        """
        assert x.dim() == 5, f"Expected 5D input (batch, time, channels, height, width), got {x.dim()}D"
        batch_size, time_steps, _, height, width = x.size()

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)

        outputs = []

        for t in range(time_steps):
            h, c = self.convlstm_cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class ResidualBlock3D(nn.Module):
    """3D Residual block for temporal-spatial feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm3d(out_channels)
            )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for residual block with optional gradient checkpointing."""
        if self.training and x.requires_grad:
            # Use gradient checkpointing during training to save memory
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class ConvLSTMEncoder(nn.Module):
    """
    Encoder component of CLVAE using ConvLSTM for temporal modeling.

    Input: (batch, 4, 16, 16, 2) → Output: μ (batch, 128), σ (batch, 128)
    """

    def __init__(self, input_channels: int = 2, hidden_channels: int = 64, latent_dim: int = 128):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        # ConvLSTM layer for temporal processing
        self.convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size=3)

        # Two residual blocks with 3D convolutions
        self.res_block1 = ResidualBlock3D(hidden_channels, 32, stride=2)
        self.res_block2 = ResidualBlock3D(32, 16, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Bottleneck dense layer
        self.bottleneck = nn.Linear(16, 8)

        # Separate dense layers for μ and σ
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 4, 16, 16, 2)
        return_features : bool
            Whether to return intermediate features for skip connections

        Returns
        -------
            Tuple of (mu, logvar) each of shape (batch, 128)
            If return_features=True, also returns list of intermediate features
        """
        # Input validation
        assert x.dim() == 5, f"Expected 5D input (batch, time, height, width, channels), got {x.dim()}D"
        assert x.shape[1:] == (4, 16, 16, 2), f"Expected shape (*, 4, 16, 16, 2), got {x.shape}"

        # Rearrange for ConvLSTM: (batch, time, channels, height, width)
        x = x.permute(0, 1, 4, 2, 3)  # (batch, 4, 2, 16, 16)

        # ConvLSTM processing
        lstm_out = self.convlstm(x)  # (batch, 4, 64, 16, 16)

        # Rearrange for 3D convolutions: (batch, channels, time, height, width)
        conv_input = lstm_out.permute(0, 2, 1, 3, 4)  # (batch, 64, 4, 16, 16)

        # Store intermediate features for skip connections
        features = []
        if return_features:
            features.append(conv_input)

        # Residual blocks
        res1_out = self.res_block1(conv_input)  # (batch, 32, 2, 8, 8)
        if return_features:
            features.append(res1_out)

        res2_out = self.res_block2(res1_out)  # (batch, 16, 1, 4, 4)
        if return_features:
            features.append(res2_out)

        # Global average pooling
        out = self.global_pool(res2_out)  # (batch, 16, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (batch, 16)

        # Bottleneck
        out = F.relu(self.bottleneck(out))  # (batch, 8)

        # μ and σ parameters
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if return_features:
            return mu, logvar, features
        return mu, logvar


class ConvLSTMDecoder(nn.Module):
    """
    Decoder component reconstructing input from latent representation.

    Input: latent vector (batch, 128) → Output: (batch, 4, 16, 16, 2)
    """

    def __init__(self, latent_dim: int = 128, hidden_channels: int = 64, output_channels: int = 2):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        # Dense layer to expand latent vector
        self.fc_expand = nn.Linear(latent_dim, 16 * 4 * 4)

        # Transpose convolution layers
        self.conv_transpose1 = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv_transpose2 = nn.ConvTranspose3d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv_transpose3 = nn.ConvTranspose3d(64, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(hidden_channels)

        # Final ConvLSTM for temporal reconstruction
        self.convlstm = ConvLSTM(hidden_channels, output_channels, kernel_size=3)

        # Final output layer
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, encoder_features: list[torch.Tensor] | None = None) -> torch.Tensor:
        """
        Forward pass for decoder with optional skip connections.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape (batch, 128)
        encoder_features : Optional[list[torch.Tensor]]
            List of encoder features for skip connections
            [conv_input, res1_out, res2_out] with shapes:
            [(batch, 64, 4, 16, 16), (batch, 32, 2, 8, 8), (batch, 16, 1, 4, 4)]

        Returns
        -------
            Reconstructed tensor of shape (batch, 4, 16, 16, 2)
        """
        # Input validation
        assert z.dim() == 2, f"Expected 2D latent vector (batch, latent_dim), got {z.dim()}D"
        batch_size = z.size(0)

        # Expand latent vector
        out = F.relu(self.fc_expand(z))  # (batch, 16*4*4)
        out = out.view(batch_size, 16, 1, 4, 4)  # (batch, 16, 1, 4, 4)

        # Transpose convolutions with optional skip connections
        out = F.relu(self.bn1(self.conv_transpose1(out)))  # (batch, 32, 2, 8, 8)

        # Add skip connection from res_block1 if available
        if encoder_features is not None and len(encoder_features) >= 2:
            skip_feat = encoder_features[1]  # res1_out: (batch, 32, 2, 8, 8)
            if skip_feat.shape == out.shape:
                out = out + skip_feat

        out = F.relu(self.bn2(self.conv_transpose2(out)))  # (batch, 64, 4, 16, 16)

        # Add skip connection from conv_input if available
        if encoder_features is not None and len(encoder_features) >= 1:
            skip_feat = encoder_features[0]  # conv_input: (batch, 64, 4, 16, 16)
            if skip_feat.shape == out.shape:
                out = out + skip_feat

        out = F.relu(self.bn3(self.conv_transpose3(out)))  # (batch, 64, 4, 16, 16)

        # Rearrange for ConvLSTM: (batch, time, channels, height, width)
        lstm_input = out.permute(0, 2, 1, 3, 4)  # (batch, 4, 64, 16, 16)

        # ConvLSTM processing
        lstm_out = self.convlstm(lstm_input)  # (batch, 4, 2, 16, 16)

        # Apply final convolution to each time step
        outputs = []
        for t in range(lstm_out.size(1)):
            frame_out = torch.sigmoid(self.final_conv(lstm_out[:, t]))
            outputs.append(frame_out.unsqueeze(1))

        output = torch.cat(outputs, dim=1)  # (batch, 4, 2, 16, 16)

        # Rearrange to match input format: (batch, 4, 16, 16, 2)
        output = output.permute(0, 1, 3, 4, 2)

        return output


class CLVAE(nn.Module):
    """
    Complete CLVAE model integrating encoder-decoder with contrastive learning.

    Total parameters: 576,395 (as specified in the plan)
    """

    def __init__(self, input_channels: int = 2, hidden_channels: int = 64, latent_dim: int = 128):
        super().__init__()

        self.latent_dim = latent_dim

        # Shared encoder and decoder
        self.encoder = ConvLSTMEncoder(input_channels, hidden_channels, latent_dim)
        self.decoder = ConvLSTMDecoder(latent_dim, hidden_channels, input_channels)

        # Loss weights (following paper's equation 1)
        self.alpha = 0.1  # KL divergence weight
        self.beta = 0.7  # Reconstruction loss weight
        # Contrastive weight = (1 - alpha - beta) = 0.2

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling: z = μ + σ × ε

        Parameters
        ----------
            mu: Mean parameters (batch, latent_dim)
            logvar: Log variance parameters (batch, latent_dim)

        Returns
        -------
            Sampled latent vector (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, use_skip_connections: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for CLVAE with optional skip connections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 4, 16, 16, 2)
        use_skip_connections : bool
            Whether to use skip connections between encoder and decoder

        Returns
        -------
            Tuple of (reconstruction, mu, logvar, latent_z)
        """
        # Encode with optional feature extraction for skip connections
        if use_skip_connections:
            mu, logvar, encoder_features = self.encoder(x, return_features=True)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z, encoder_features)
        else:
            mu, logvar = self.encoder(x, return_features=False)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z)

        return reconstruction, mu, logvar, z

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        contrastive_pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """
        Compute three loss components: reconstruction, KL divergence, and contrastive.

        Parameters
        ----------
        x : torch.Tensor
            Original input
        reconstruction : torch.Tensor
            Reconstructed output
        mu : torch.Tensor
            Mean parameters
        logvar : torch.Tensor
            Log variance parameters
        contrastive_pairs : tuple[torch.Tensor, torch.Tensor] | None
            Optional tuple of (pre_flood_mu, post_flood_mu) for contrastive learning

        Returns
        -------
        dict
            Dictionary containing individual losses and total loss
        """
        batch_size = x.size(0)

        # Reconstruction loss (MSE for continuous SAR data)
        recon_loss = F.mse_loss(reconstruction, x, reduction="sum") / batch_size

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Contrastive loss (if pairs provided)
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if contrastive_pairs is not None:
            pre_mu, post_mu = contrastive_pairs

            # Cosine similarity between pre-flood and post-flood latent representations
            # We want to MAXIMIZE the distance between pre/post flood states in latent space
            # This encourages the model to learn distinct representations for normal vs flood conditions
            cos_sim = F.cosine_similarity(pre_mu, post_mu, dim=1)

            # Loss = 1 - cosine_similarity (paper uses cosine similarity loss)
            # When cos_sim = 1 (identical), loss = 0 (bad, we want them different)
            # When cos_sim = -1 (opposite), loss = 2 (we're pushing them apart, good)
            # When cos_sim = 0 (orthogonal), loss = 1 (decent separation)
            # Note: Paper uses weight 0.2 for contrastive loss vs 0.7 for reconstruction
            contrastive_loss = torch.mean(1 - cos_sim)

        # Total loss (following paper's equation 1)
        # L_Total = α*L_KL + β*L_Recon + (1-α-β)*L_Contrast
        contrastive_weight = 1 - self.alpha - self.beta  # = 0.2
        total_loss = self.alpha * kl_loss + self.beta * recon_loss + contrastive_weight * contrastive_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "contrastive_loss": contrastive_loss,
        }

    def encode(
        self, x: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Encode input to latent parameters."""
        return self.encoder(x, return_features=return_features)

    def decode(self, z: torch.Tensor, encoder_features: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z, encoder_features)
