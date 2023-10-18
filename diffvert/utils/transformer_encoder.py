""" Encoder flax module used for track embeddings. """
import jax.numpy as jnp
from jax.config import config

from flax import linen as nn
config.update("jax_enable_x64", True)

class TransformerEncoder(nn.Module):
    """ Transformer Encoder used for track data.

    Follows pre-attention transformer encoder structure.
    Encoder runs tracks through a few dense layers and then a variable input number of self
    attention layers mixed with denser layers and normalization.

    Attributes:
        num_attention_layers: how many self attention layers to use for encoding.
        num_attention_heads: how many heads are used in multi-head self-attention
    """
    num_attention_layers: int
    num_attention_heads: int

    @nn.compact
    def __call__(self, tracks):
        embd = nn.Sequential([
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.sigmoid,
        ])(tracks)

        x = embd

        for _ in range(self.num_attention_layers):
            n = nn.SelfAttention(num_heads=self.num_attention_heads, param_dtype=jnp.float64)(x)
            x = x + n
            x = nn.LayerNorm(param_dtype=jnp.float64)(x)
            n = nn.Dense(features=128, param_dtype=jnp.float64)(x)
            n = nn.relu(n)
            n = nn.Dense(features=32, param_dtype=jnp.float64)(n)
            x = x + n
            x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        return x
