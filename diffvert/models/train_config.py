""" Module containing information used for training different versions of the model.

Contains all hyperparameters for model training as well as options for different portions of the
model, such as activation functions and vertexing strategy.

Options that require more than two possibilities are turned into enums to avoid strings in the
model code.
"""
import jax_dataclasses as jdc
from enum import IntEnum

class WeightActivation(IntEnum):
    """ Specify activation function for track weights. """
    SOFTMAX = 0
    SIGMOID = 1
    PERFECT_WEIGHTS = 2
    NO_TRACK_SELECTION = 3


class Vertexer(IntEnum):
    """ Specify how the vertex is reconstructed (and if it is). """
    NDIVE = 0
    NONE = 1
    TRUE_VERTEX = 2


@jdc.pytree_dataclass(frozen=True)
class TrainConfig:
    """ Store hyperparameters and configurations (e.g. activation types) for training. """

    model_name: str = "NDIVE"
    num_epochs: int = 300
    samples: str = "/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/all_flavors"
    batch_size: int = 100
    learning_rate: float = 1e-5
    pretrained_NDIVE: bool = False

    track_weight_activation: int = int(WeightActivation.SIGMOID)
    num_attention_layers: int = 1
    num_attention_heads: int = 1

    jet_flavor_loss: bool = True
    track_origin_loss: bool = True
    track_pairing_loss: bool = True
    vertex_loss: bool = True
    use_mse_loss: bool = False
    normalize_vertex_loss: bool = False
    chi_squared_loss: bool = False
    track_weight_loss: bool = False

    vertexer: int = int(Vertexer.NDIVE)

    use_ghost_track: bool = True
    clip_vertex: bool = False

    use_one_hot_encoding: bool = False
    use_early_stopping: bool = False
    use_adam: bool = False
    use_cosine_decay_schedule: bool = False
    use_learning_rate_decay_when_stalled: bool = False

    config_name: str = "none" # how this config will be saved

