"""Hyena Hierarchies."""

from jax import numpy as jnp

from flax import linen as nn

from einops import rearrange
from opt_einsum import contract

import gin

from utils import types


class Sin(nn.Module):
    """Sinusoidal layer."""

    dim: int
    omega: float = 10.0
    trainable = True

    def setup(self):
        """Initialize the layer."""
        self.freq = self.param(
            "freq", nn.initializers.ones, (1, self.dim,)) if self.trainable \
            else jnp.ones((1, self.dim,))

    def __call__(self, x):
        """Apply the layer."""
        return jnp.sin(self.omega * x * self.freq)


def get_positional_embedding_initializer(emb_dim: int, seq_len: int):
    """Get the positional embedding initializer."""

    def _initilizer(key: types.PRNGKey, shape: types.Unused = None):
        """Initialize the positional embedding."""
        t = jnp.linspace(0., 1., seq_len)[None, :, None]

        bands = (emb_dim - 1) // 2

        t_rescaledd = jnp.linspace(0., 1.*seq_len - 1., seq_len)[None, :, None]
        w = 2. * jnp.pi * t_rescaledd / seq_len     # (1, seq_len, 1)

        f = jnp.linspace(1e-4, bands - 1., bands)[None, None]
        z = jnp.exp(-1j * w * f)    # (1, seq_len, bands)

        z = jnp.concatenate([t, jnp.real(z), jnp.imag(z)], axis=-1)

        return z    # (1, seq_len, emb_dim)

    return _initilizer


def get_exponential_module_initializer(
        d_model: int,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        shift: float = 0.0,

):
    """Get the exponential module initializer."""

    def _initilizer(key: types.PRNGKey, shape: types.Unused = None):
        """Initialize the exponential module."""
        max_decay = jnp.log(target) / fast_decay_pct
        min_decay = jnp.log(target) / slow_decay_pct
        deltas = jnp.linspace(min_decay, max_decay, d_model)[None, None]
        return deltas

    return _initilizer


@gin.configurable
class HyenaFilter(nn.Module):
    """Hyena filter layer."""

    d_model: int
    emb_dim: int = 3
    filter_order: int = 16
    seq_len: int = 1024
    dropout_rate: float = 0.0
    omega: float = 1.0
    n_mlps: int = 2
    exp_mod_shift = 0.

    def setup(self):
        """Initialize the layer."""
        self.bias = self.param(
            'bias', nn.initializers.normal(), (self.d_model,))
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # positional embedding ------------------------------------------------
        assert self.emb_dim % 2 != 0 and self.emb_dim >= 3, \
            "emb_dim must be odd and >= 3 (time, sine, cosine)"
        self.pos_emb_z = self.param(
            'pos_emb_z',
            get_positional_embedding_initializer(self.emb_dim, self.seq_len),
            (1, self.seq_len, self.emb_dim)
        )
        self.pos_emb_t = jnp.linspace(0., 1., self.seq_len)[None, :, None]
        # ---------------------------------------------------------------------

        # exponential modulation ----------------------------------------------
        self.exp_mod_deltas = self.param(
            'exp_mod_deltas', get_exponential_module_initializer(self.d_model))
        self.exp_mod_shift = jnp.array(self.exp_mod_shift)
        # ---------------------------------------------------------------------

        # MLPs ---------------------------------------------------------------
        act = Sin(dim=self.filter_order, omega=self.omega)
        self.implicit_filter = nn.Sequential([
            nn.Sequential([nn.Dense(self.filter_order), act])
            for _ in range(self.n_mlps)
        ] + [nn.Dense(self.d_model, use_bias=False)])
        # ---------------------------------------------------------------------

    def get_filters(self, training: bool = False):
        """Get the Hyena filter."""
        z, t = self.pos_emb_z, self.pos_emb_t
        h = self.implicit_filter(z)
        h = self.dropout(h, deterministic=not training)
        decay = jnp.exp(jnp.abs(self.exp_mod_deltas) * -t) + self.exp_mod_shift
        h = h * decay
        return h[0]

    def __call__(self, x, h, b):
        """Do fft conv + bias.

        Args:
            x: input tensor
            h: filter
            b: bias

        Returns:
            output tensor
        """
        fft_size = self.seq_len * 2
        h_f = jnp.fft.rfft(h, n=fft_size) / float(fft_size)
        x_f = jnp.fft.rfft(x, n=fft_size)

        y = jnp.fft.irfft(
            h_f * x_f, n=fft_size, norm='forward')[..., :self.seq_len]
        y = y + x * b[..., None]

        return y


@gin.configurable
class Hyena(nn.Module):
    """Hyena model."""

    order: int = 2
    dropout_rate: float = 0.1
    filter_dropout_rate: float = 0.0
    l_max: int = gin.REQUIRED   # changed manually per task in gin config
    d_model: int = gin.REQUIRED
    filter_order: int = gin.REQUIRED

    def setup(self):
        """Initialize the model."""
        inner_width = self.d_model * (self.order + 1)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.in_proj = nn.Dense(inner_width)
        self.out_proj = nn.Dense(self.d_model)

        self.short_filter = nn.Conv(
            features=inner_width,
            kernel_size=[3],
            padding=2,
            feature_group_count=inner_width,
        )

        self.hyena_filter = HyenaFilter(
            d_model=self.d_model * (self.order - 1),
            filter_order=self.filter_order,
            seq_len=self.l_max,
            dropout_rate=self.filter_dropout_rate,
        )

    def __call__(self,
                 u: jnp.ndarray, training: bool = False, *args, **kwargs):
        """Forward pass."""
        l_input = u.shape[-2]
        l_filter = min(l_input, self.l_max)
        u = self.in_proj(u)

        uc = self.short_filter(u)[..., :l_filter]
        uc = rearrange(u, 'b l d -> b d l')
        *x, v = uc.split(self.order + 1, axis=1)

        # get filters and biases
        h = self.hyena_filter.get_filters(training=training)
        h = rearrange(h, 'l (o d) -> o d l', o=self.order-1)
        b = self.hyena_filter.bias
        b = rearrange(b, '(o d) -> o d', o=self.order-1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i, deterministic=not training)
            v = self.hyena_filter(v, h=h[o], b=b[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y
