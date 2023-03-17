"""Stack blocks into a final model."""

from flax import linen as nn

import gin


@gin.configurable
class StackedModel(nn.Module):
    """A model is a stack of blocks."""

    d_output: int
    n_blocks: int = 4
    d_model: int = gin.REQUIRED
    block: nn.Module = gin.REQUIRED

    def setup(self):
        """Initialize the model."""
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.blocks = [
            self.block(index=idx)
            for idx in range(self.n_blocks)
        ]

    def __call__(self, x, training=False):
        """Run the model."""
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, training)
        x = self.decoder(x)
        return x
