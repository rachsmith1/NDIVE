""" Test data format util module.

Tests helper functions in the module.
"""
import numpy as np
import diffvert.utils.data_format as daf
import torch

class TestMasking:
    """ Test the masks created by data format based on num tracks in a jet. """

    def test_basic_mask(self):
        """ Tests mask over 'num_jets' x 'max_num_tracks' """
        num_tracks = 7
        dummy_tracks = np.zeros((1, 15, 51))
        dummy_tracks[0,:,daf.JetData.N_TRACKS] = num_tracks
        mask = daf.create_tracks_mask(dummy_tracks)
        for idx in range(15):
            assert mask[0][idx] == (idx < num_tracks)

        mask_ghost = daf.create_tracks_mask(dummy_tracks, pad_for_ghost=True)
        assert(mask_ghost.shape[1] == 16)
        for idx in range(16):
            assert mask_ghost[0][idx] == (idx < num_tracks+1)


    def test_track_pair_mask(self):
        """ Tests mask over 'num_jets' x 'max_num_tracks' x 'max_num_tracks' """
        num_tracks = 6
        dummy_tracks = np.zeros((1, 15, 51))
        dummy_tracks[0,:, daf.JetData.N_TRACKS] = num_tracks
        mask = daf.create_tracks_mask(dummy_tracks)
        pair_mask = daf.create_track_pairs_mask(mask)

        for idx_1 in range(15):
            for idx_2 in range(15):
                assert pair_mask[0][idx_1][idx_2] == (idx_1 < num_tracks and idx_2 < num_tracks)
