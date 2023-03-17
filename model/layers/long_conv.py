"""Long Convolutions Layer."""

from jax import numpy as jnp

from flax import linen as nn

from einops import rearrange
from opt_einsum import contract

import gin


@gin.configurable
class LongConv(nn.Module):
    """Long Convolutions Layer."""

    n_channels: int = 4
    dropout_rate: float = 0.1
    l_max: int = gin.REQUIRED   # changed manually per task in gin config
    d_model: int = gin.REQUIRED

    def setup(self):
        """Set up the layer."""
        self.in_proj = nn.Dense(self.d_model)
        self.out_proj = nn.Dense(self.d_model)
        self.kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (self.n_channels, self.d_model, self.l_max))
        self.dmat = self.param(
            'dmat',
            nn.initializers.lecun_normal(),
            (self.n_channels, self.d_model))

    def __call__(self, u: jnp.ndarray, training=False) -> jnp.ndarray:
        """Forward pass."""
        u = self.in_proj(u)     # x: (batch_size, seq_len, d_model)
        # reshape x to have seq_len as last dimension
        u = rearrange(u, 'b l d -> b d l')  # x: (batch_size, d_model, seq_len)
        length = u.shape[-1]

        # do fft conv
        k_f = jnp.fft.rfft(self.kernel, n=2*length)
        u_f = jnp.fft.rfft(u, n=2*length)
        y_f = contract('bdl,cdl->bcdl', u_f, k_f)
        y = jnp.fft.irfft(y_f, n=2*length)[..., :length]

        # add skip connection
        # TODO(mahanf): resolve with skip con in SeqBlock
        y += contract('bdl,cd->bcdl', u, self.dmat)

        # flatten and reorder
        y = rearrange(y, '... c d l -> ... l (c d)')

        y = self.out_proj(y)
        return y
