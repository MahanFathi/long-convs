"""Stack layers into a block."""

import functools

from flax import linen as nn

import gin


@gin.configurable
class SequenceResidualBlock(nn.Module):
    """A block of layers with residual connection."""

    index: int = None
    layer: nn.Module = gin.REQUIRED
    d_model: int = gin.REQUIRED
    dropout_rate: float = 0.0

    def setup(self):
        """Set up the block."""
        self.seq = self.layer()
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0])

    def __call__(self, x, training=False):
        """Forward pass."""
        x2 = self.seq(x)
        drop = functools.partial(self.drop, deterministic=not training)
        z = drop(self.out(drop(nn.gelu(x2))))
        return self.norm(z + x)
