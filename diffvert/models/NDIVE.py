""" Module containing NDIVE (differentiable vertexing) flax module. 

NDIVE network takes in jets of tracks and outputs a vertex position. The given tracks are grouped
by jet but not necessarily by vertex. The network predicts a track's importance to the value of
the vertex. These weights are adjusted by leveraging a differentiable vertex fitter used by the
model.

Contains: NDIVE network, config to model function, and loss function
"""
import jax
import jax.numpy as jnp
from jax.config import config

from flax import linen as nn

from diffvert.utils.billoir_vertex_fit import billoir_vertex_fit
from diffvert.utils.transformer_encoder import TransformerEncoder
import diffvert.utils.data_format as daf

import diffvert.models.train_config as tc

config.update("jax_enable_x64", True)


class Network(nn.Module):
    """ NDIVE network for predicting decay vertex from jets. 
    
    Attributes:
        track_weight_activation: from tc.WeightActivation. Activation function to be used on
            track weights (may also be instruction to use truth value weights)
        num_attention_layers: how many attention layers to use in track embedding pre-vertexing
        use_ghost_track: whether or not to add a ghost track in jet direction to each jet
        clip_vertex: whether or not to clip vertex at large values (to aid in training)
    """
    track_weight_activation: int
    num_attention_layers: int
    num_attention_heads: int
    use_ghost_track: bool
    clip_vertex: bool

    def create_ghost_track(self, tracks):
        """ Create ghost track in jet direction for batched jets.

        Args:
            tracks: 'num_jets' x 'max_num_tracks' x 'num_track_params' arrays of inputs
        Returns:
            'num_jets' x 1 x 'num_track_inputs' array of ghost tracks
        """
        num_jets, max_num_tracks = tracks.shape[0:2]
        return jnp.stack([
            jnp.array([
                0.,
                0.,
                0.,
                phi,
                theta,
                0.,
                0.,
                0.,
                0.,
                jnp.mean(d_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(z_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(p_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(t_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                jnp.mean(r_o, where=jnp.arange(0, max_num_tracks, dtype=jnp.int32) < n),
                0.,
                0.,
                jnp.log(pt),
                eta,
            ]) for n,phi,theta,d_o,z_o,p_o,t_o,r_o,pt,eta in zip(
                tracks[:,0,daf.JetData.N_TRACKS],
                tracks[:,0,daf.JetData.TRACK_JET_PHI],
                tracks[:,0,daf.JetData.TRACK_JET_THETA],
                tracks[:,:,daf.JetData.TRACK_D0_ERR],
                tracks[:,:,daf.JetData.TRACK_Z0_ERR],
                tracks[:,:,daf.JetData.TRACK_PHI_ERR],
                tracks[:,:,daf.JetData.TRACK_THETA_ERR],
                tracks[:,:,daf.JetData.TRACK_RHO_ERR],
                tracks[:,0,daf.JetData.TRACK_JET_PT],
                tracks[:,0,daf.JetData.TRACK_JET_ETA],
            )
        ]).reshape(num_jets,1,daf.NUM_JET_INPUT_PARAMETERS)


    @nn.compact
    def __call__(self, tracks, key):
        """ Predicts vertex from raw track data.

        Args:
            tracks: num_jets x n_trks x n_track params track data
            key: jax prng key
        Returns:
            vertex_fit: num_jets x 3 cartesian coordinates of vertex prediction
            vertex_covariance_fit: num_jets x 3 x 3 covariance of vtx prediction
            track_in_vertex_weights: num_jets x n_tracks computed weight of track for fit
            vertex_fit_chi2: array of chi2 from each jet's fit
        """
        num_jets, max_num_tracks = tracks.shape[0:2]

        n_trks = tracks[:, 0, daf.JetData.N_TRACKS]

        if self.use_ghost_track:
            # pad by one to account for addition of ghost track
            n_trks += 1
            max_num_tracks += 1

        # mask padding: size = (num_jets, max_num_tracks)
        mask = daf.create_tracks_mask(tracks, pad_for_ghost=self.use_ghost_track)

        mask_tracks = jnp.repeat(
            mask, daf.NUM_JET_INPUT_PARAMETERS,
        ).reshape(num_jets, max_num_tracks, daf.NUM_JET_INPUT_PARAMETERS)

        orig_tracks = tracks

        if self.use_ghost_track:
            # add ghost track in the jet trajectory
            ghost_track = self.create_ghost_track(tracks)

            # tracks = tracks[:,:,0:daf.NUM_JET_INPUT_PARAMETERS] # remove truth value data
            tracks = daf.get_track_inputs(tracks)
            tracks = jnp.concatenate((ghost_track, tracks), axis=1)
        else:
            tracks = daf.get_track_inputs(tracks)

        # embed tracks with transformer, predict their contribution to vertex
        masked_track_data = jnp.where(mask_tracks == 0, 0, tracks)
        track_embeddings = TransformerEncoder(
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
        )(
            masked_track_data
        )
        track_in_vertex_weights = nn.Sequential(
            [
                nn.Dense(features=32, param_dtype=jnp.float64),
                nn.relu,
                nn.Dense(features=1, param_dtype=jnp.float64),
            ]
        )(track_embeddings).reshape(num_jets, max_num_tracks)

        # use activation on predicted weights for tracks affecting decay vertex
        track_in_vertex_weights = jnp.where(mask == 0, -jnp.inf, track_in_vertex_weights)

        if self.track_weight_activation == tc.WeightActivation.SOFTMAX:
            track_in_vertex_weights = nn.softmax(track_in_vertex_weights, axis=1)

        if self.track_weight_activation == tc.WeightActivation.SIGMOID:
            track_in_vertex_weights = nn.sigmoid(track_in_vertex_weights)

        if self.track_weight_activation == tc.WeightActivation.PERFECT_WEIGHTS:
            # perfect weights are 1 if from hadron decay vertex, 0 o.w.
            perfect_weights = jnp.array(
                abs(orig_tracks[:,:,daf.JetData.TRACK_PROD_VTX_Z]
                    - jnp.repeat(
                        orig_tracks[:,0,daf.JetData.HADRON_Z],
                        max_num_tracks,axis=0
                    ).reshape(num_jets,max_num_tracks)
                ) < 1e-3,
                dtype=jnp.float64
            )
            # set true weight for ghost track to be 1
            if self.use_ghost_track:
                perfect_weights = jnp.concatenate(
                    (jnp.ones((num_jets,1),dtype=jnp.float64), perfect_weights),
                    axis=1
                )
            track_in_vertex_weights = perfect_weights
            
        if self.track_weight_activation == tc.WeightActivation.NO_TRACK_SELECTION:
            track_in_vertex_weights = jnp.where(mask != 0, 1, track_in_vertex_weights)

        track_in_vertex_weights = jnp.where(mask == 0, 1e-100, track_in_vertex_weights)

        # do vertex fit with weights
        vertex_fit, vertex_covariance_fit, vertex_fit_chi2 = billoir_vertex_fit(
            tracks,
            track_in_vertex_weights,
            jnp.zeros((num_jets, 3)), # vertex seeded to be the origin
        )

        # fix nans, potentially from masking issues
        if self.clip_vertex:
            vertex_fit = jnp.clip(vertex_fit, a_min=-4000.0, a_max=4000.0)
        vertex_fit = jax.numpy.nan_to_num(
            vertex_fit, nan=4000.0, posinf=4000.0, neginf=-4000.0,
        )
        vertex_covariance_fit = jax.numpy.nan_to_num(
            vertex_covariance_fit, nan=1000.0, posinf=1000.0, neginf=1000.0,
        )
        vertex_fit_chi2 = jax.numpy.nan_to_num(
            vertex_fit_chi2, nan=1000.0, posinf=1000.0, neginf=1000.0,
        )

        vertex_fit = vertex_fit.reshape(num_jets, 3)
        vertex_covariance_fit = jax.lax.stop_gradient(vertex_covariance_fit).reshape(num_jets, 3, 3)
        vertex_fit_chi2 = jax.lax.stop_gradient(vertex_fit_chi2).reshape(num_jets, 1)

        return vertex_fit, vertex_covariance_fit, track_in_vertex_weights, vertex_fit_chi2


def model_from_config(cfg: tc.TrainConfig):
    """ create NDIVE module with functions as specified in config 
    
    Args:
        cfg: TrainConfig specifying module operations
    Returns:
        instance of NDIVE class with operations as specified in cfg
    """
    return Network(
        track_weight_activation=cfg.track_weight_activation,
        num_attention_layers=cfg.num_attention_layers,
        num_attention_heads=cfg.num_attention_heads,
        use_ghost_track=cfg.use_ghost_track,
        clip_vertex=cfg.clip_vertex,
    )


def loss_function(ytrue, xtrue, outputs, cfg: tc.TrainConfig):
    """ compute loss for NDIVE. outputs vertex only

    Args:
        ytrue: truth value outputs
        xtrue: full truth value inputs (contains true decay vertex)
        cfg: TrainConfig
    Returns:
        total loss, and tuple of individual computed losses
    """
    vertex_fit = outputs[0]

    # euclidean distance loss for vertex position
    vertex_pred = vertex_fit.reshape(-1, 3)
    vertex_true = xtrue[:, 0, daf.JetData.HADRON_X:daf.JetData.HADRON_Z+1].reshape(-1, 3)
    loss_euclidean_distance = jnp.sqrt(jnp.sum((vertex_true - vertex_pred) ** 2, axis=1))
    loss_euclidean_distance = jnp.mean(loss_euclidean_distance)

    # mean absolute error loss for vertex position
    loss_mean_abs_err = abs(vertex_true - vertex_pred)
    loss_mean_abs_err = jnp.mean(loss_mean_abs_err)

    loss_total = loss_mean_abs_err
    if cfg.use_mse_loss:
        loss_total = loss_euclidean_distance

    return loss_total, (loss_mean_abs_err, loss_euclidean_distance)
