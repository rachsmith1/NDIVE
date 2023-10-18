""" Module containing information on how data is formatting, and functions to deal with format.

As input to the jax model, the track data is completely flattened. This module allows
access to the data through more understandable enums.
"""
from enum import IntEnum
import jax.numpy as jnp

# NUM_JET_INPUT_PARAMETERS: int = 16
NUM_JET_INPUT_PARAMETERS: int = 18  # count used as inputs in flavor tagging

class JetData(IntEnum):
    """ Store which indices of track inputs mean what. """
    TRACK_PT = 0
    TRACK_D0 = 1
    TRACK_Z0 = 2
    TRACK_PHI = 3
    TRACK_THETA = 4
    TRACK_RHO = 5
    TRACK_PT_FRACTION_LOG = 6  # log(track_pt / jet_pt)
    TRACK_DELTA_R = 7  # deltaR(track, jet)
    TRACK_PT_ERR = 8
    TRACK_D0_ERR = 9
    TRACK_Z0_ERR = 10
    TRACK_PHI_ERR = 11
    TRACK_THETA_ERR = 12
    TRACK_RHO_ERR = 13
    TRACK_SIGNED_SIG_D0 = 14  # signed d0 significance
    TRACK_SIGNED_SIG_Z0 = 15  # signed z0 significance
    # begin true production vertex info
    TRACK_PROD_VTX_X = 16
    TRACK_PROD_VTX_Y = 17
    TRACK_PROD_VTX_Z = 18
    # begin coordinates of true hadron decay for b,c jets ((0,0,0) otherwise)
    HADRON_X = 19
    HADRON_Y = 20
    HADRON_Z = 21
    # begin jet info
    N_TRACKS = 22
    TRACK_VERTEX_INDEX = 23
    # begin jet-level variables (really should be renamed to only be JET_X not TRACK_JET_X)
    TRACK_JET_PHI = 24
    TRACK_JET_THETA = 25
    TRACK_JET_PT = 26
    TRACK_JET_ETA = 27
    TRACK_JET_FLAVOR = 28
    # 29-43 binary track-level variables for training the vertex pairs auxiliary task
    # 44-46 binary jet-level variables for training the jet flavor task
    # 47-50 binary track-level variables for training the track origin auxiliary task
    TRACK_FROM_B = 47
    TRACK_FROM_C = 48
    TRACK_FROM_ORIGIN = 49
    TRACK_FROM_OTHER = 50


class JetPrediction(IntEnum):
    """ Store which indices of JetPrediction mean what. """
    PROB_B = 0
    PROB_C = 1
    PROB_U = 2
    VERTEX_X = 3
    VERTEX_Y = 4
    VERTEX_Z = 5
    VERTEX_COV_XX = 6
    VERTEX_COV_XY = 7
    VERTEX_COV_XZ = 8
    VERTEX_COV_YX = 9
    VERTEX_COV_YY = 10
    VERTEX_COV_YZ = 11
    VERTEX_COV_ZX = 12
    VERTEX_COV_ZY = 13
    VERTEX_COV_ZZ = 14
    VERTEX_TRACK_STARTS = 15


def get_track_inputs(tracks):
    """ Return inpust from full track data.
    
    Args:
        tracks: 'num_jets' x 'max_num_tracks' x 'num_track_params' input tracks
    Returns:
        values used as inputs (specifically avoids truth-value data) of shape
            'num_jets' x 'max_num_tracks' x 'NUM_JET_INPUT_PARAMETERS'
    """
    # return tracks[:,:,0:JetData.TRACK_PROD_VTX_X]

    # add jet-level pt, eta to track-level params as done in gn1
    return jnp.concatenate(
        (
            tracks[:,:,0:JetData.TRACK_PROD_VTX_X], # all single-track, non-truth variables
            jnp.log(tracks[:,:,JetData.TRACK_JET_PT:JetData.TRACK_JET_PT+1]),
            tracks[:,:,JetData.TRACK_JET_ETA:JetData.TRACK_JET_ETA+1],
        ),
        axis=2,
    )


def create_tracks_mask(tracks, pad_for_ghost=False):
    """ Create a mask of size 'num_jets' x 'max_num_tracks' for which tracks are real.
    
    Args:
        tracks: 'num_jets' x 'max_num_tracks' x 'num_track_params' input tracks
        pad_for_ghost: whether or not to pad for later inclusion of ghost track
    Returns:
        boolean mask indicating if tracks are real or padding ('num_jets' x 'max_num_tracks')
    """
    num_jets, max_num_tracks, = tracks.shape[0:2]
    max_num_tracks += pad_for_ghost

    # each jet has track indices 0, 1, 2, ... max_num_tracks-1
    track_indices = jnp.tile(
        jnp.arange(0,max_num_tracks,dtype=jnp.int32),
        num_jets,
    ).reshape(num_jets, max_num_tracks)

    # 'num_jets' x 'max_num_tracks' array of real tracks in each jet (repeated)
    n_trks = jnp.repeat(
        tracks[:,0,JetData.N_TRACKS]+pad_for_ghost, max_num_tracks,
    ).reshape(num_jets, max_num_tracks)

    mask =  jnp.where(track_indices < n_trks, 1, 0)
    return mask


def create_track_pairs_mask(mask):
    """ Create a mask of size 'num_jets' x 'max_num_tracks' x 'max_num_tracks' for track pairs.

    A track pair is valid iff both tracks are valid.
    
    Args:
        mask: 'num_jets' x 'max_num_tracks' boolean mask array
    Returns:
        'num_jets' x 'max_num_tracks' x 'max_num_tracks' boolean track pair mask
    """
    num_jets, max_num_tracks = mask.shape

    mask_first_track = jnp.repeat(
        mask, max_num_tracks,
    ).reshape(num_jets, max_num_tracks, max_num_tracks)

    mask_second_track = jnp.repeat(
        mask.reshape(num_jets, 1, max_num_tracks), max_num_tracks, axis=1,
    ).reshape(num_jets, max_num_tracks, max_num_tracks)

    mask_track_pairs = jnp.logical_and(mask_first_track, mask_second_track)
    return mask_track_pairs

