""" Module containing smaller networks used for auxiliary tasks.

Auxiliary tasks predict a property of tracks or jets from track and/or jet embeddings.
"""
import jax.numpy as jnp
from flax import linen as nn


class FlavorPredictionNetwork(nn.Module):
    """ Module for predicting probability jets are b/c/u jets.
    
    Output is probability distribution over three possible jet flavors.
    """
    @nn.compact
    def __call__(self, jet_embedding):
        """ Predicts flavors from batched jet embeddings. """
        num_jets = jet_embedding.shape[0]

        raw_flavor_pred = nn.Sequential([
            nn.Dense(features=64, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=16, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=3, param_dtype=jnp.float64),
        ])(jet_embedding)

        # flavor prediction is prob b/c/u jet
        raw_flavor_pred = jnp.reshape(raw_flavor_pred, (num_jets, 3))
        flavor_pred = nn.softmax(raw_flavor_pred, axis=1)
        return flavor_pred


class TrackOriginPredictionNetwork(nn.Module):
    """ Module for predicting track origin as from b decay, c decay, primary vertex, or other.

    Output is probability distribution over four possible track origins. 
    """
    @nn.compact
    def __call__(self, jet_embedding, track_embeddings):
        """ Predict track origins for batched track data. """
        num_jets, max_num_tracks = track_embeddings.shape[0:2]

        # append jet embedding to each track for prediction
        jet_embedding_repeated = jnp.repeat(
            jet_embedding, max_num_tracks, axis=0,
        ).reshape(num_jets, max_num_tracks, -1)

        full_track_embedding = jnp.concatenate((track_embeddings, jet_embedding_repeated), axis=2)

        # take track embeddings to dimension 4 prediction of origin from b/c/origin/other
        track_origin_pred = nn.Sequential([
            nn.Dense(features=64, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=16, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=4, param_dtype=jnp.float64),
        ])(full_track_embedding)

        track_origin_pred = nn.softmax(track_origin_pred, axis=2)
        return track_origin_pred


class TrackPairingPredictionNetwork(nn.Module):
    """ Module for predicting if each pair of tracks is from the same vertex/origin.
    
    Output is probability that the pair is from the same origin.
    """
    @nn.compact
    def __call__(self, jet_embedding, track_embeddings):
        """ Predict pairing from batched jet and track embeddings. """

        num_jets, max_num_tracks = track_embeddings.shape[0:2]

        # full embedding for a pair is concatenation of two track embeddings, jet embedding
        first_track_embedding = jnp.repeat(track_embeddings, max_num_tracks, axis=1)
        second_track_embedding = jnp.repeat(
            track_embeddings, max_num_tracks, axis=0,
        ).reshape(num_jets, max_num_tracks**2, -1)

        jet_embedding_trk_pairs = jnp.repeat(
            jet_embedding, max_num_tracks**2, axis=0,
        ).reshape(num_jets, max_num_tracks**2, -1)

        track_pairing_pred = jnp.concatenate(
            (first_track_embedding, second_track_embedding, jet_embedding_trk_pairs),
            axis=2,
        )

        # MLP input dim is 'num_jets' x 'max_num_tracks ^ 2' x '2*trk_embed + jet_embed'
        track_pairing_pred = nn.Sequential([
            nn.Dense(features=64, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=32, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=16, param_dtype=jnp.float64),
            nn.relu,
            nn.Dense(features=1, param_dtype=jnp.float64),
        ])(track_pairing_pred)

        track_pairing_pred = jnp.reshape(
            track_pairing_pred,
            (num_jets, max_num_tracks, max_num_tracks),
        )
        track_pairing_pred = nn.sigmoid(track_pairing_pred)
        return track_pairing_pred
