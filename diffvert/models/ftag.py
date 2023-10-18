""" Flavor Tagging model, supporting different configurations.

Main model of the project. Instances are constructed from model_from_config file.
Does end-to-end track to flavor prediction, taking into account multiple auxiliary tasks as
specified in config to aid in training.

Contains: flavor tagging network, config to model function, and loss function.

Follows closely GN1 architecture: https://cds.cern.ch/record/2811135
"""
import jax
import jax.numpy as jnp
from jax.config import config
from flax import linen as nn

from diffvert.utils.track_extrapolation import track_extrapolation
from diffvert.utils.transformer_encoder import TransformerEncoder
import diffvert.models.auxiliary_task_networks as auxnets

from diffvert.models.NDIVE import Network as NDIVE
import diffvert.models.train_config as tc
import diffvert.utils.data_format as daf

config.update("jax_enable_x64", True)


class Network(nn.Module):
    """ Flavor tagging network.
    
    Takes in track data, outputs jet flavor predcition along with information for auxiliary tasks.
    """
    track_weight_activation: int
    num_attention_layers: int
    num_attention_heads: int
    vertexer: str
    use_ghost_track: bool
    use_one_hot_encoding: bool
    clip_vertex: bool

    def append_one_hot_encoding(self, orig_tracks, mask, tracks, inverse_order=True):
        """ Add one hot encoding to track data.

        Args:
            orig_tracks: original track inputs
            mask: mask of where tracks are valid (not from padding)
            tracks: 'num_jets' x 'max_num_tracks' x 'num_track_inps'
                tracks to append one hot encoding to
            inverse_order: whether or not to invert tracks pre-one-hot
        Returns:
            tracks appended with one hot encoding on pt
                'num_jets' x 'max_num_tracks' x 'num_track_inps + max_num_tracks'
        """
        _, max_num_tracks, _ = orig_tracks.shape

        # create one hot encoding on order of track_pt by jet
        track_pt = orig_tracks[:,:,daf.JetData.TRACK_PT]
        track_pt = jnp.where(mask, track_pt, 0)
        track_pt_ids = jnp.argsort(track_pt, axis=1)

        if inverse_order:
            track_pt_ids = track_pt_ids[:, ::-1]

        one_hots = jax.nn.one_hot(track_pt_ids, max_num_tracks)
        return jnp.concatenate((tracks, one_hots), axis=2)


    @nn.compact
    def __call__(self, tracks, key):
        """ Run flavor tagging model for batched track data. 
        
        Args:
            tracks: num_jetx x n_trks x n_trk params input data
            key: jax prng key
        Returns:
            flavor: num_jets x 3 probabilities over b/c/u
            track_origin_pred: num_jets x 4 probabilities over b/c/origin/other
            track_pairing_pred: num_jets x n_trks x n_trks prob from same origin
            vertex_fit, cov_fit, chi2_fit: vertex fit output for each jet
            track_in_vertex_weights: num_jets x num_tracks weights for fit
        """
        num_jets, max_num_tracks, _ = tracks.shape

        # mask-out padded tracks: 'num_jets' x 'max_num_tracks'
        mask = daf.create_tracks_mask(tracks)

        mask_tracks = jnp.repeat(
            mask, daf.NUM_JET_INPUT_PARAMETERS,
        ).reshape(num_jets, max_num_tracks, daf.NUM_JET_INPUT_PARAMETERS)

        mask_origin_pred = jnp.repeat(
            mask, 4,
        ).reshape(num_jets, max_num_tracks, 4)

        mask_track_inputs = jnp.repeat(
            mask, daf.NUM_JET_INPUT_PARAMETERS+3, # appends vertex to each track
        ).reshape(num_jets, max_num_tracks, daf.NUM_JET_INPUT_PARAMETERS+3)

        # shape: 'num_jets' x 'max_num_tracks' x 'max_num_tracks'
        mask_track_pairs = daf.create_track_pairs_mask(mask)

        # instantiate vertex fit to all zeros, so we can return same values for all configs
        vertex_fit = jnp.zeros((num_jets, 3))
        cov_fit = jnp.zeros((num_jets, 3, 3))
        track_in_vertex_weights = jnp.zeros((num_jets, max_num_tracks+self.use_ghost_track))
        chi2_fit = jnp.zeros((num_jets))

        if self.vertexer == tc.Vertexer.NDIVE:
            vertex_fit, cov_fit, track_in_vertex_weights, chi2_fit = NDIVE(
                track_weight_activation=self.track_weight_activation,
                num_attention_layers=self.num_attention_layers,
                num_attention_heads=self.num_attention_heads,
                use_ghost_track=self.use_ghost_track,
                clip_vertex=self.clip_vertex,
            )(
                tracks, key
            )
        if self.vertexer == tc.Vertexer.TRUE_VERTEX:
            vertex_fit = tracks[:,0,daf.JetData.HADRON_X:daf.JetData.HADRON_Z+1]

        # mask tracks before encoder, clip off truth value information for inputs
        track_inputs = daf.get_track_inputs(tracks)
        track_inputs = jnp.where(mask_tracks == 0, 0, track_inputs)

        # NDIVE, Track extrapolation
        if self.vertexer == tc.Vertexer.NDIVE or self.vertexer == tc.Vertexer.TRUE_VERTEX:
            v = vertex_fit.reshape(num_jets, 3)
            extrapolated_tracks = track_extrapolation(
                tracks, v,
            ).reshape(num_jets, max_num_tracks, 8)[:, :, 3:8] # get five perigee params (after vtx)
            # full extrapolated track data is same as original data but with perigee params changed
            #   append fit vertex (location for new perigee parameters)
            t_extp = jnp.concatenate(
                (
                    track_inputs[:, :, daf.JetData.TRACK_PT].reshape(num_jets, max_num_tracks, 1),
                    extrapolated_tracks,
                    track_inputs[:, :, daf.JetData.TRACK_RHO+1:],
                    jnp.repeat(v, max_num_tracks, axis=0).reshape(num_jets, max_num_tracks, 3),
                ),
                axis=2,
            )
            t_extp = jnp.where(mask_track_inputs == 0, 0, t_extp)

        # append the perigee vertex to track information (orig tracks around origin)
        t_orig = jnp.concatenate((track_inputs, jnp.zeros((num_jets, max_num_tracks, 3))), axis=2)
        t_orig = jnp.where(mask_track_inputs == 0, 0, t_orig)

        # encode original tracks
        track_processor = TransformerEncoder(
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
        )

        if self.use_one_hot_encoding:
            t_orig = self.append_one_hot_encoding(track_inputs, mask, t_orig)

        t_orig_encoded = track_processor(t_orig)
        track_embeddings = t_orig_encoded

        # encode extrapolated tracks if there is a vertexer
        if self.vertexer == tc.Vertexer.NDIVE or self.vertexer == tc.Vertexer.TRUE_VERTEX:
            if self.use_one_hot_encoding:
                t_extp = self.append_one_hot_encoding(
                    track_inputs, mask, t_extp, inverse_order=False,
                )
            t_extp_encoded = track_processor(t_extp)
            track_embeddings = jnp.concatenate((t_orig_encoded, t_extp_encoded), axis=2)

        # Graph Attention Pooling to create jet embedding from all track embeddings
        jet_embedding_track_weights = nn.softmax(
            jnp.where(
                mask == 0,
                -jnp.inf,
                nn.Dense(features=1, param_dtype=jnp.float64, name="JetEmbeddingPoolingDense")(
                    track_embeddings,
                ).reshape(num_jets, max_num_tracks),
            ),
            axis=1,
        )

        jet_embedding = jnp.einsum(
            "bij,bjk->bik",
            jet_embedding_track_weights.reshape(num_jets, 1, max_num_tracks),
            track_embeddings,
        ).reshape(num_jets, -1)

        # use track and jet embeddings to compute all tasks (flavor and auxiliary)
        flavor_pred = auxnets.FlavorPredictionNetwork()(jet_embedding)

        track_origin_pred = auxnets.TrackOriginPredictionNetwork()(
            jet_embedding, track_embeddings,
        )
        track_origin_pred = jnp.where(mask_origin_pred == 0, 0, track_origin_pred)

        track_pairing_pred = auxnets.TrackPairingPredictionNetwork()(
            jet_embedding, track_embeddings,
        )
        track_pairing_pred = jnp.where(mask_track_pairs == 0, 0, track_pairing_pred)

        return (flavor_pred, track_origin_pred, track_pairing_pred,
                vertex_fit, cov_fit, track_in_vertex_weights, chi2_fit,)


def model_from_config(cfg: tc.TrainConfig):
    """ Contruct an ftag model given a training config.

    Args:
        cfg: config used to specify module variables
    Returns:
        ftag model instance as specified by cfg
    """
    return Network(
        track_weight_activation=cfg.track_weight_activation,
        num_attention_layers=cfg.num_attention_layers,
        num_attention_heads=cfg.num_attention_heads,
        vertexer=cfg.vertexer,
        use_ghost_track=cfg.use_ghost_track,
        use_one_hot_encoding=cfg.use_one_hot_encoding,
        clip_vertex=cfg.clip_vertex,
    )


def binary_cross_entropy(ytrue, ypred, weights=None):
    """ Binary cross entropy, supports weighting (may or may not be batched). """
    ypred = jnp.clip(ypred, a_min=1e-6, a_max=1.0 - 1e-6)
    loss = -ytrue * jnp.log(ypred) - (1.0 - ytrue) * jnp.log(1 - ypred)

    if weights is not None:
        loss = loss * weights

    return loss


def categorical_cross_entropy(ytrue, ypred, weights):
    """ Categorical cross entropy loss function. Supports weighting and assumes input follows
    'batch_dim' x data_dim format, where data_dim is a tuple of length 2.

    Args:
        ytrue: true outputs 'num_jets' x data_dim
        ypred: predicted outputs 'num_jets' x data_dim
        weights: optional weighting over inputs in batch
    Returns:
        'num_jets' x data_dim[0] of cross entropy values summed over last dimension
    """
    ypred = jnp.clip(ypred, a_min=1e-6, a_max=1.0 - 1e-6)
    loss = -ytrue * jnp.log(ypred)

    if weights is not None:
        loss = loss * weights

    return jnp.sum(loss, axis=2)


def loss_function(ytrue, xtrue, outputs, cfg: tc.TrainConfig):
    """ Compute loss of outputs for an ftag model given config of which losses to include.

    total loss is summed over which losses are included by the config
    all possible losses are always computed and returned even if nonsensical. if not in the config
        they are not factored into the total loss and thus not included in the gradient

    Args:
        ytrue: truth values for input batch (TODO: add formatting info)
        xtrue: full JetData input (contains truth values), of dimension
               'num_jets' x 'max_n_tracks' x 'num_track_params'
        outputs: predicted values for input batch
        cfg: config used for model. relevant piece here is inclusion of loss functions
    Returns:
        total loss, tuple of individual loss function outs (includes value for all possible, 
            even if some excluded by config)
    """
    (flavor_pred, track_origin_pred, track_pairing_pred, vertex_fit,
     cov_fit, track_in_vertex_weights, chi2_fit) = outputs

    num_jets, max_num_tracks = xtrue.shape[0:2]

    true_track_pairing = ytrue[:, :, 3:3+max_num_tracks]
    true_jet_flavor = ytrue[:, 0, 3+max_num_tracks:3+max_num_tracks+3]
    true_track_origin = ytrue[:, :, 3+max_num_tracks+3:3+max_num_tracks+3+4]

    vertex_true = xtrue[:, 0, daf.JetData.HADRON_X:daf.JetData.HADRON_Z+1]
    vertex_pred = vertex_fit.reshape(-1, 3)

    # mask 'num_jets' x 'max_num_tracks'
    mask = daf.create_tracks_mask(xtrue)

    # mask 'num_jets' x 'max_num_tracks' x 'max_num_tracks'
    mask_track_pairs = daf.create_track_pairs_mask(mask)

    # jet flavor loss from output prob of b/c/u versus one-hot encoding of true values
    loss_jet_flavor = jnp.reshape(
        categorical_cross_entropy(
            jnp.reshape(true_jet_flavor, (num_jets, 1, 3)),
            jnp.reshape(flavor_pred, (num_jets, 1, 3)),
            None,
        ),
        (num_jets, 1),
    )
    loss_jet_flavor = jnp.mean(loss_jet_flavor)

    # track origin loss from track-wise output prob of from b/c/origin/other versus one-hot of truth
    loss_track_origin = jnp.reshape(
        categorical_cross_entropy(true_track_origin, track_origin_pred, None),
        (num_jets, max_num_tracks, 1),
    )
    loss_track_origin = jnp.mean(loss_track_origin, where=mask.reshape(num_jets,max_num_tracks,1))

    # track pairing loss from output prob pairwise tracks from same source vs binary of truth
    loss_track_pairing = jnp.reshape(
        binary_cross_entropy(true_track_pairing, track_pairing_pred),
        (num_jets, max_num_tracks, max_num_tracks),
    )
    loss_track_pairing = jnp.mean(loss_track_pairing, where=mask_track_pairs)

    # vertex fit loss from MAE for truth vertex - predicted vertex (all cartesian coordinates)
    loss_vertex_fit = jnp.abs(vertex_true - vertex_pred)
    loss_vertex_fit_mse = jnp.sqrt(jnp.sum((vertex_true-vertex_pred)**2, axis=1))
    if cfg.normalize_vertex_loss:
        # set normalization to true decay length
        normalization = jnp.sqrt(jnp.sum(vertex_true**2, axis=1))
        normalization = jnp.maximum(normalization, 1.0)  # deal with u decay length 0
        loss_vertex_fit = loss_vertex_fit / jnp.repeat(normalization, 3).reshape(num_jets,3)
        loss_vertex_fit_mse = loss_vertex_fit_mse / normalization

    loss_vertex_fit = jnp.mean(loss_vertex_fit)
    loss_vertex_fit_mse = jnp.mean(loss_vertex_fit_mse)
    if cfg.use_mse_loss:
        loss_vertex_fit = loss_vertex_fit_mse

    # chi2 loss is just the chi2 of the vertex fit (does not use truth data)
    loss_chi2 = jnp.mean(chi2_fit)

    # track weight loss is from output track weights versus truth of if from hadron vertex
    perfect_weights = jnp.array(
        abs(xtrue[:,:,daf.JetData.TRACK_PROD_VTX_Z]
            - jnp.repeat(xtrue[:,0,daf.JetData.HADRON_Z],15,axis=0).reshape(-1,15)) < 1e-3,
        dtype=jnp.float64
    )

    # ignore ghost track for track weight losses
    if cfg.use_ghost_track:
        track_in_vertex_weights = track_in_vertex_weights[:,1:]
    loss_track_weight = binary_cross_entropy(track_in_vertex_weights, perfect_weights)
    loss_track_weight = jnp.mean(loss_track_weight, where=mask)

    # add up losses for all included losses, as dictated by the config
    loss_total = 0.0
    if cfg.jet_flavor_loss:
        loss_total += loss_jet_flavor
    if cfg.track_origin_loss:
        loss_total += 0.5 * loss_track_origin
    if cfg.track_pairing_loss:
        loss_total += 1.5 * loss_track_pairing
    if cfg.vertex_loss:
        loss_total += 0.1 * loss_vertex_fit
    if cfg.chi_squared_loss:
        loss_total += .001 * loss_chi2
    if cfg.track_weight_loss:
        loss_total += loss_track_weight

    return loss_total, (loss_jet_flavor, loss_track_origin, loss_track_pairing,
                        loss_vertex_fit, loss_chi2, loss_track_weight)
