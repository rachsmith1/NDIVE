""" Library of functions to make plotting model details more convenient. 

Handles common patterns in plotting (e.g. plotting info for all flavors separately)
"""
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
from dataclasses import asdict

import os

import flax
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import optax

import torch

from functools import partial

import diffvert.utils.data_format as daf
from diffvert.models.train_config import TrainConfig

import importlib

import matplotlib.pyplot as plt

MAX_NUM_TRACKS = 15
MODEL_INPUT_SHAPE = [10,MAX_NUM_TRACKS,51]

def full_save_path(save_dir, epoch=None, model_number=None):
    """ Get the path to a model by saved config name, epoch, and model number. """
    path = f"{os.getenv('NDIVE_MODEL_PATH')}{save_dir}"
    if model_number is not None:
        path += f"/model_{model_number}"
    if epoch is not None:
        path += f"/{epoch}"
    return path


def get_param_count(save_dir:str, epoch:int|None=None, model_number=None):
    """ Get param dictionary and total param count for a saved model. """
    full_save_dir = full_save_path(save_dir, epoch, model_number)
    # initialize model with checkpointed parameters
    restored_vals = checkpoints.restore_checkpoint(
        ckpt_dir=full_save_dir, target=None, step=0, parallel=False
    )
    print(full_save_dir)

    config_dict = asdict(TrainConfig())
    config_dict.update(restored_vals["config"])
    cfg: TrainConfig = TrainConfig(*config_dict.values())
    print(f"cfg: {cfg}")

    # set model parameters to checkpointed values
    params = restored_vals["model"]["params"]
    # print(params.keys())
    params = freeze(params)

    # create a model state for inference. optimizer is irrelevant
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    return param_count, params


def get_test_output(
        save_dir:str,
        epoch:int|None=None,
        model_number:int|None=None,
        ignore_saved_file:bool=False
    ):
    """ return output on standard test data for save dir and epoch 

    Note: suggested to use non-None epoch because default may be over-written by new training
    
    Args:
        save_dir: string name of the directory we care for.
                  equivalent to 'config_name' in TrainConfig
        epoch: (optional) epoch of training to look at model.
        model_number: (optional) int indicating model number (used for training copies)
        ignore_saved_file: if true, recompute outputs, overwriting cached outs if they exist
    """
    full_save_dir = full_save_path(save_dir, epoch, model_number)
    output_save_path = full_save_dir+"/test_outputs.npz"

    if not ignore_saved_file and os.path.exists(output_save_path):
        print("found previous cached outputs")
        loaded_arrays = np.load(output_save_path)
        return loaded_arrays
    else:
        print("could not find cached outputs, running inference")

    device_count = jax.device_count()
    print(device_count)
    test_vmap_count = 50
    print(test_vmap_count)
    jax.devices()

    b_samples_dir = "/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/bjets"
    c_samples_dir = "/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/cjets"
    u_samples_dir = "/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/ujets"
    b_test_dl = torch.load(f"{b_samples_dir}/test_dl.pth")
    c_test_dl = torch.load(f"{c_samples_dir}/test_dl.pth")
    u_test_dl = torch.load(f"{u_samples_dir}/test_dl.pth")

    # initialize model with checkpointed parameters
    restored_vals = checkpoints.restore_checkpoint(
        ckpt_dir=full_save_dir, target=None, step=0, parallel=False
    )
    print(full_save_dir)

    # load config dict from defaults and update with saved value to allow retroactivity
    config_dict = asdict(TrainConfig())
    config_dict.update(restored_vals["config"])
    cfg: TrainConfig = TrainConfig(*config_dict.values())
    print(f"cfg: {cfg}")

    model_from_config = getattr(
        importlib.import_module(f"diffvert.models.{cfg.model_name}"),
        "model_from_config",
    )
    model = model_from_config(cfg)

    # set model parameters to checkpointed values
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.ones(MODEL_INPUT_SHAPE), rng)["params"]
    params = unfreeze(params)
    params = restored_vals["model"]["params"]
    # print(params.keys())
    params = freeze(params)

    # create a model state for inference. optimizer is irrelevant
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.novograd(0.1),
    )
    state_dist = flax.jax_utils.replicate(state)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"loaded model parameter count: {param_count}")

    key = jax.random.PRNGKey(23)

    @partial(jax.pmap, axis_name="device", in_axes=(0, 0, None), out_axes=(0, 0, 0, 0, 0, 0, 0))
    def eval_step_pmap(state, batch_x, k):

        def eval_step(state, batch_x):
            x1, x2, x3, v1, e1, w1, y = model.apply({"params": state.params}, batch_x, k)
            return x1, x2, x3, v1, e1, w1, y

        eval_vmap = jax.vmap(eval_step, in_axes=(None, 0), out_axes=(0, 0, 0, 0, 0, 0, 0))
        x1, x2, x3, v1, e1, w1, y = eval_vmap(state, batch_x)

        return x1, x2, x3, v1, e1, w1, y

    @partial(jax.pmap, axis_name="device", in_axes=(0, 0, None), out_axes=(0, 0, 0, 0))
    def eval_step_pmap_NDIVE(state, batch_x, k):

        def eval_step(state, batch_x):
            v1, e1, w1, y = model.apply({"params": state.params}, batch_x, k)
            return v1, e1, w1, y

        eval_vmap = jax.vmap(eval_step, in_axes=(None, 0), out_axes=(0, 0, 0, 0))
        v1, e1, w1, y = eval_vmap(state, batch_x)

        return v1, e1, w1, y

    def single_inference(state, inp, only_NDIVE=False):
        """ do inference for a single batch, returning outputs of model as one array
        
        Args:
            state: model state
            inp: batched data of dimensions 'n_batch' x 'input_dims'
            only_NDIVE: whether model is just NDIVE or full end-to-end
        Returns:
            jnp array containing output of inference of dimensions 'n_batch' x 'output_dims'
        """
        x_batch = jax.tree_map(
            lambda m: m.reshape((device_count, test_vmap_count, -1, *m.shape[1:])),
            inp,
        )
        if not only_NDIVE:
            x1, x2, x3, v1, e1, w1, y = eval_step_pmap(state, x_batch, key)
        else:
            v1, e1, w1, y = eval_step_pmap_NDIVE(state, x_batch, key)
            x1 = jnp.zeros(v1.shape) # create zeros to match the format as full model

        x1 = jnp.reshape(x1, (-1,3))
        v1 = jnp.reshape(v1, (-1,3))
        e1 = jnp.reshape(e1, (-1,9))
        w1 = jnp.reshape(w1, (-1,MAX_NUM_TRACKS+cfg.use_ghost_track)) # +1 for ghost track

        return jnp.concatenate((x1, v1, e1, w1), axis=1)

    def evaluate(state, data_loader, only_NDIVE=False):
        """ do inference over entirety of data loader 
        
        Args:
            state: model state
            data_loader: torch data loader object containing data
            only_NDIVE: whether model is a full end-to-end model or just NDIVE. purpose
                        is that NDIVE-only gives a subset of the proper outputs
        Returns:
            input_meta_arr: jnp array containing jet-wise input info 'n_jets' x 'num_jet_params'
            output_arr: jnp array containing model output for jet-flavour and vertex location 
                        'n_jets' x 'output_dim'
        """

        input_list = []
        output_list = []
        for d in data_loader:
            outs  = single_inference(state_dist, jnp.array(d.x), only_NDIVE)
            input_list.append(jnp.array(d.x))
            output_list.append(outs)
        input_arr = jnp.concatenate(input_list)
        output_arr = jnp.concatenate(output_list)
        print(f"final input shape: {input_arr.shape}")
        print(f"final output shape: {output_arr.shape}")
        return input_arr, output_arr

    b_input_arr, b_output_arr = evaluate(
        state_dist, b_test_dl, only_NDIVE=(cfg.model_name=="NDIVE"),
    )
    c_input_arr, c_output_arr = evaluate(
        state_dist, c_test_dl, only_NDIVE=(cfg.model_name=="NDIVE"),
    )
    u_input_arr, u_output_arr = evaluate(
        state_dist, u_test_dl, only_NDIVE=(cfg.model_name=="NDIVE"),
    )

    np.savez(
        output_save_path,
        b_input_arr=b_input_arr,
        b_output_arr=b_output_arr,
        c_input_arr=c_input_arr,
        c_output_arr=c_output_arr,
        u_input_arr=u_input_arr,
        u_output_arr=u_output_arr,
    )

    return np.load(output_save_path)


def get_test_output_list(
        save_dirs: list[str],
        epochs: list[int|None]|None=None,
        model_numbers: list[int|None]|None=None,
        ignore_saved_file:bool=False,
    ):
    """ Get multiple outputs at once. 
    
    Used for multiple trainings of same model.
    """
    output_list = []
    if epochs is None:
        epochs = [None]*len(save_dirs)
    for save_dir, epoch, model_number in zip(save_dirs, epochs, model_numbers):
        output_list.append(get_test_output(save_dir, epoch, model_number, ignore_saved_file))
    return output_list


# bunch of graphing functions
def graph_output_info(
    fn,
    ax,
    outs_list,
    ins_list,
    colors=["#1f77b4", "#ff7f0e", "#2ca02c"], #plt defaults
    labels=["b","c","u"],
    linestyles=["solid","solid","solid"],
    nan_val: float | None =None,
    plot_avgs=True,
    has_ghost=True,
    hist_options=dict(),
):
    """ graph histogram of jet-wise output data on ax for all flavors
    
    Args:
        fn: function to extract data from 'num_jets' x ('num_outs', 'num_ins') arrays
        ax: plt figure axis
        b/c/u_outs: outputs for b, c, u jets
        b/c/u_ins: jet input data for b, c, u jets
        nan_val: value (or None) to turn nan's into for say graphing percent of time purity is 0/0
        plot_avgs: whether or not to include line for average
        has_ghost: whether or not prediciton includes ghost track output
        hist_options: dict of extra histogram options
    Returns:
        None
    Modifies:
        ax, by adding relevant histograms
    """
    hist_options.update(dict(histtype="step"))

    for idx, (outs, ins) in enumerate(zip(outs_list, ins_list)):
        processed_outs = fn(outs, ins, has_ghost=has_ghost)
        if nan_val is not None: processed_outs = np.nan_to_num(processed_outs, nan=nan_val)
        ax.hist(
            processed_outs,
            label=labels[idx],
            color=colors[idx],
            weights=np.repeat(1/len(processed_outs), len(processed_outs)),
            linestyle=linestyles[idx],
            **hist_options,
        )
        if plot_avgs:
            ax.axvline(
                np.mean(processed_outs),
                color=colors[idx],
                linestyle="--",
                alpha=0.3,
            )
    ax.legend()


def graph_output_info_dict(fn, ax, outs, plot_avgs=False, labels=["b","c","u"], linestyles=["solid","solid","solid"], has_ghost=True, hist_options=None):
    """ graph histogram of jet-wise output data on ax for all flavors
    
    Args:
        fn: function to extract data from 'num_jets' x ('num_outs', 'num_ins') arrays
        ax: plt figure axis
        outs: dictionary containing inference outputs and inputs in saved format
        has_ghost: whether or not ghost is included
    """
    graph_output_info(
        fn,
        ax,
        outs_list=[outs["b_output_arr"],outs["c_output_arr"],outs["u_output_arr"]],
        ins_list=[outs["b_input_arr"],outs["c_input_arr"],outs["u_input_arr"]],
        plot_avgs=plot_avgs,
        labels=labels,
        linestyles=linestyles,
        has_ghost=has_ghost,
        hist_options=hist_options,
    )

    
def graph_output_info_vs_input_info(
    output_fn,
    input_fn,
    ax,
    #xax,
    bins,
    outs_list,
    ins_list,
    colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    labels=["b","c","u"],
    markers=[","],
    linestyles=["solid"],
    dofillarea=False,
    doboxes=False,
    horizontal_shift=0,
    drop_y_err=False,
    outs_vs_outs=False,
    has_ghost=True,
    add_legend=True,
    name_extrafig=None,
    errortype="std"
):
    """ graph function on the output vs function on the input. must be jet-wise
    
    Args:
        output_fn: must take in both outputs and inputs in that order
        input_fn: takes in only inputs or outputs and inputs depending on 'outs_vs_outs'
        ax: axis to plot on
        bins: bins for x-axis. contains leftmost and rightmost boundaries
        b/c/u outs: outputs by flavor
        b/c/u ins: inputs by flavor
        outs_vs_outs: true if graphing two output types against each other
    """

    def bin_data(inputs, outputs):
        bin_indices = np.digitize(inputs, bins, right=True)-1
        in_bins = []
        out_bins = []
        errdown_bins = []
        errup_bins = []
        print("Using %s prescription"%errortype)
        for idx in range(len(bins)-1):
            in_bins.append(inputs[bin_indices==idx])
            if len(outputs)==2: # this is ugly but just means that the function I used return two values per jet instead of one; used for eff or purity.
                k = np.sum(outputs[0][bin_indices==idx])
                N = np.sum(outputs[1][bin_indices==idx])
                eff = k/N
                edown, eup = errors_on_efficiency(k,N,errortype)
                out_bins.append(eff)
                errdown_bins.append(edown)
                errup_bins.append(eup)
                #print("bin ", idx, ": eff, error down, error up = ", eff, edown, eup)
            else: out_bins.append(outputs[bin_indices==idx])
        return in_bins, out_bins, errdown_bins, errup_bins

    def plot_binned(processed_ins, processed_outs, args={}):
        
        xs = [(a+b)/2 for a, b in zip(bins, bins[1:])]
        xerrs = [(b-a)/2 for a, b in zip(bins, bins[1:])]

        in_bins, out_bins, errdown_bins, errup_bins = bin_data(processed_ins, processed_outs)

        # make some diagnostic plots
        if name_extrafig is not None:
            figtmp, xax = plt.subplots(1, len(bins)-1,figsize=(25, 5),layout="constrained")
            xax[0].set_title(name_extrafig)
            for i,xx in enumerate(xax): xx.hist(out_bins[i],bins=10)
            #plt.tight_layout()
            plt.savefig("%s.pdf"%name_extrafig)
            plt.close(figtmp)
        
        ys = [np.nanmean(out_bin) for out_bin in out_bins]
        if len(errdown_bins)>0: 
            yerrs = np.array([(np.nanmean(d),np.nanmean(u)) for d,u in zip(errdown_bins,errup_bins)]).T
        else: yerrs = [np.nanstd(out_bin) for out_bin in out_bins]
        if drop_y_err: yerrs=None
        
        if dofillarea:
            ax.fill_between(xs, ys-yerrs[0], ys+yerrs[1], linewidth=0, alpha=0.3, color=args['color'])
            ax.plot(xs, ys, **args)
            ax.set_xticks(bins)
        elif doboxes: 
            ax.boxplot(out_bins, 0, '', positions=list(map(lambda x,e: x + e, bins[:-1], xerrs)), widths=xerrs, medianprops=dict(color=args['color'],linewidth=2))
        else:
            args['linestyle']='none'
            ax.errorbar(list(map(lambda x: x + horizontal_shift, xs)), ys, xerr=xerrs, yerr=yerrs, fmt='', capsize=0, markersize=7, linewidth=2, **args)
            ax.set_xticks(bins)

    for idx, (outs, ins) in enumerate(zip(outs_list, ins_list)):
        processed_outs = output_fn(outs, ins, has_ghost=has_ghost)
        if outs_vs_outs:
            processed_ins = input_fn(outs, ins, has_ghost=has_ghost)
        else:
            processed_ins = input_fn(ins)

        plot_binned(processed_ins, processed_outs, args=dict(label=labels[idx], linestyle=linestyles[idx], color=colors[idx], marker=markers[idx]))
    if add_legend: ax.legend()
    

def graph_stacked_bar(
        output_fns, input_fn, ax, bins,
        jet_outs=None, jet_ins=None, output_names=None, colors=None,
    ):
    """ graph stacked bar chart on the output vs function on the input. must be jet-wise
    
    Args:
        output_fns: list of functions to stack
        input_fn: function on the inputs (must take both outs and ins)
        ax: axis to plot on
        bins: list of bin sides (includes both leftmost and rightmost borders)
        jet_outs: outputs of function
    """

    def bin_data(inputs, outputs):
        """ bin based on inputs """
        bin_indices = np.digitize(inputs, bins, right=True)-1
        in_bins = []
        out_bins = []
        for idx in range(len(bins)-1):
            in_bins.append(inputs[bin_indices==idx])
            out_bins.append(outputs[bin_indices==idx])
        return in_bins, out_bins

    bar_heights = [] # list of 'num out functions' x 'num bins' containing pure values
    in_counts = np.zeros((len(bins)-1))
    for output_fn in output_fns:
        processed_ins = input_fn(jet_ins)
        processed_outs = output_fn(jet_outs)
        in_bins, out_bins = bin_data(processed_ins, processed_outs)
        in_lens = np.array([len(in_bin) for in_bin in in_bins])
        in_counts = in_lens
        bar_heights.append(np.array([np.sum(out_bin) for out_bin in out_bins]))

    bottom = np.zeros((len(bins)-1))
    bin_mids = [(a+b)/2 for a, b in zip(bins, bins[1:])]
    bin_width = bins[1]-bins[0] # would need to change if not even bins
    for idx, bar_set in enumerate(bar_heights):
        label = idx if output_names is None else output_names[idx]
        color = colors[idx] if colors is not None else None
        ax.bar(
            bin_mids, bar_set/in_counts, width=bin_width,
            label=label, bottom=bottom, color=color,
        )
        bottom += bar_set/in_counts
    ax.legend()


# bunch of functions on track outputs and inputs
def get_tracks_above_threshold(
        outputs, threshold=0.5, remove_ghost_track=True,
        include_ghost_norm=True, has_ghost=True,
    ):
    """ return binary array of size 'n_jets' x 'n_tracks' containing whether 
    each track is above threshold * max weight
    
    Args:
        outputs: 'n_jets' x 'n_outs' array of outputs
        threshold: percentage of max weight for which track is considered identified from decay
        remove_ghost_track: whether or not to remove ghost track from final data
        include_ghost_norm: whether or not to remove weight of ghost track when computing max weight
        has_ghost: whether or not output has ghost track out
    """
    weights = outputs[:,daf.JetPrediction.VERTEX_TRACK_STARTS:daf.JetPrediction.VERTEX_TRACK_STARTS + MAX_NUM_TRACKS+has_ghost]
    if include_ghost_norm or not has_ghost:
        normal_weights = weights / np.max(weights, axis=1).reshape(-1,1)
    else:
        normal_weights = weights / np.max(weights[:,1:], axis=1).reshape(-1,1)
    above_threshold = normal_weights > threshold

    if remove_ghost_track and has_ghost:
        above_threshold = above_threshold[:,1:]

    return above_threshold


def get_tracks_from_decay(inputs):
    """ return binary array of size 'n_jets' x 'n_tracks' indicating whether the track
    should be counted as from the decay vertex
    """
    from_decay = np.logical_or(
        inputs[:,:,daf.JetData.TRACK_FROM_B].astype(int)==1,
        inputs[:,:,daf.JetData.TRACK_FROM_C].astype(int)==1,
    )
    return from_decay


def get_ratio_true_pos_true(outputs, inputs, threshold=0.5, has_ghost=True):
    """ Get efficiency of track selection: 
        percent of tracks which should be included that are included; 
    """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost)

    from_decay = get_tracks_from_decay(inputs)
    num_should_be_included = np.sum(from_decay, axis=1)
    print("\n Calculating Efficiency...")    
    print(f"num should be shape: min: {np.min(num_should_be_included)}, max: {np.max(num_should_be_included)}")
    num_above_and_should_be = np.sum(np.logical_and(tracks_above_threshold, from_decay), axis=1)
    
    #percent_correct = np.divide(num_above_and_should_be, num_should_be_included)
    k = num_above_and_should_be
    N = num_should_be_included
    percent_correct = k / N
    
    print(f"  nan count in efficiency calculation: {np.count_nonzero(np.isnan(percent_correct))} -> {np.count_nonzero(np.isnan(percent_correct))/percent_correct.shape[0] * 100:.2f}%")
    print(f" --- Num jets without tracks from decay: {num_should_be_included.shape[0]-np.count_nonzero(num_should_be_included)}" )
        
    return percent_correct

def get_ratio_true_pos_true_mod(outputs, inputs, threshold=0.5, has_ghost=True):
    """ Get efficiency of track selection: 
        percent of tracks which should be included that are included; 
        return numerator and denominator
    """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost)

    from_decay = get_tracks_from_decay(inputs)
    num_should_be_included = np.sum(from_decay, axis=1)
    print("\n Calculating Efficiency...")    
    print(f"num should be shape: min: {np.min(num_should_be_included)}, max: {np.max(num_should_be_included)}")
    num_above_and_should_be = np.sum(np.logical_and(tracks_above_threshold, from_decay), axis=1)
    
    k = num_above_and_should_be
    N = num_should_be_included
    percent_correct = k / N
    
    print(f"  nan count in efficiency calculation: {np.count_nonzero(np.isnan(percent_correct))} -> {np.count_nonzero(np.isnan(percent_correct))/percent_correct.shape[0] * 100:.2f}%")
    print(f" --- Num jets without tracks from decay: {num_should_be_included.shape[0]-np.count_nonzero(num_should_be_included)}" )    
    
    return k, N


def get_num_false_positive(outputs, inputs, threshold=0.5, has_ghost=True):
    """ Get count of tracks included which shouldn't be. """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost)

    not_from_decay = np.logical_not(get_tracks_from_decay(inputs))
    bad_included = np.logical_and(tracks_above_threshold, not_from_decay)
    num_bad_included = np.sum(bad_included, axis=1)
    return num_bad_included


def get_num_true_positive(outputs, inputs, threshold=0.5, has_ghost=True):
    """ Get count of tracks included which should be. """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost)

    from_decay = get_tracks_from_decay(inputs)
    good_included = np.logical_and(tracks_above_threshold, from_decay)
    num_good_included = np.sum(good_included, axis=1)
    return num_good_included


def get_num_true_positive_plus_ghost(outputs, inputs, threshold=0.5, has_ghost=True):
    """ Get count of tracks included which should be, including ghost if included. """
    tracks_above_threshold = get_tracks_above_threshold(
        outputs, threshold, remove_ghost_track=False, has_ghost=has_ghost,
    )

    from_decay = get_tracks_from_decay(inputs)
    good_included = np.logical_and(tracks_above_threshold[:,int(has_ghost):], from_decay)
    num_good_included = np.sum(good_included, axis=1)

    if has_ghost: return np.add(num_good_included, tracks_above_threshold[:,0])
    return num_good_included


def get_ratio_true_pos_pos(outputs, inputs, threshold=0.5, has_ghost=False):
    """ Get percent of tracks included which should have been (purity) """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost, include_ghost_norm=False)

    from_decay = get_tracks_from_decay(inputs)
    
    num_above_and_should_be = np.sum(np.logical_and(tracks_above_threshold, from_decay), axis=1)
    num_included = np.sum(tracks_above_threshold, axis=1)
    
    #percent = np.divide(num_above_and_should_be, num_included) # m / N
    k = num_above_and_should_be
    N = num_included
    percent = k / N

    print("\n Calculating purity...")
    print(f"  nan count in purity calculation: {np.count_nonzero(np.isnan(percent))} -> {np.count_nonzero(np.isnan(percent))/percent.shape[0] * 100:.2f}%")
    print(f" --- Num no tracks with w > 0.5 (zero count in denominator): {num_included.shape[0]-np.count_nonzero(num_included)}")
    
    #if errortype is not None: 
    #    print(f" ... and now calculating errors.")
    #    err_do, err_up = errors_on_efficiency( k, N, errortype)
    #    return percent, err_do, err_up
        
    return percent

def get_ratio_true_pos_pos_mod(outputs, inputs, threshold=0.5, has_ghost=False):
    """ Get percent of tracks included which should have been (purity) 
            return numerator and denominator
    """
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost, include_ghost_norm=False)

    from_decay = get_tracks_from_decay(inputs)
    
    num_above_and_should_be = np.sum(np.logical_and(tracks_above_threshold, from_decay), axis=1)
    num_included = np.sum(tracks_above_threshold, axis=1)
    
    #percent = np.divide(num_above_and_should_be, num_included) # m / N
    k = num_above_and_should_be
    N = num_included
    percent = k / N
    
    print("\n Calculating purity...")
    print(f"  nan count in purity calculation: {np.count_nonzero(np.isnan(percent))} -> {np.count_nonzero(np.isnan(percent))/percent.shape[0] * 100:.2f}%")
    print(f" --- Num no tracks with w > 0.5 (zero count in denominator): {num_included.shape[0]-np.count_nonzero(num_included)}")    
        
    return k,N


def errors_on_efficiency(k, N, errortype):
    """ Get errors on efficiency, using different prescriptions """

    if errortype == "std":
        # standard deviation
        err_do = np.divide(1,N)*np.sqrt(k*(1-(np.divide(k,N))))
        err_up = err_do
    elif errortype == "CP":
        err_do, err_up = clopper_pearson_interval(k, N, 0.683)
        eff = k/N
        err_do = eff - err_do
        err_up = err_up - eff
    elif errortype == None: 
        err_do = np.zeros_like(percent)
        err_up = err_do

    return err_do, err_up

def clopper_pearson_interval(k, n, confidence):
    # k : number of successes
    # n : number of trials
    # confidence level 
    from scipy.stats import beta 
    
    alpha = (1-confidence) / 2
    err_do = np.where( k==0, 0, beta.ppf(alpha, k, n-k+1))
    err_up = np.where( k==n, 1, beta.ppf(1-alpha, k+1, n-k))
    
    return err_do, err_up


def get_num_positive(outputs, inputs, threshold=0.5, remove_ghost_track=True, has_ghost=True):
    """ Get number of tracks included by model """
    tracks_above_threshold = get_tracks_above_threshold(
        outputs, threshold, remove_ghost_track=remove_ghost_track, has_ghost=has_ghost,
    )
    num_tracks_included = np.sum(tracks_above_threshold, axis=1)
    return num_tracks_included


def get_vertex_error_euclidean(outputs, inputs, has_ghost=True):
    """ Get euclidean distance from predicted vertex to true vertex. """
    true_vertices = inputs[:,0,daf.JetData.HADRON_X:daf.JetData.HADRON_Z+1]
    pred_vertices = outputs[:,daf.JetPrediction.VERTEX_X:daf.JetPrediction.VERTEX_Z+1]
    euclid_distances = np.sqrt(np.sum(
        np.square(true_vertices-pred_vertices), axis=1,
    ))
    return euclid_distances


def get_vertex_error_euclidean_normalized(outputs, inputs, has_ghost=True):
    """ Get normalized euclidean distance from predicted vertex to true vertex. """
    euclid_distnaces = get_vertex_error_euclidean(outputs, inputs)
    true_vertices = inputs[:,0,daf.JetData.HADRON_X:daf.JetData.HADRON_Z+1]
    normalization = np.sum(np.square(true_vertices), axis=1)
    normalization = np.maximum(normalization, 1.0)
    return euclid_distnaces / normalization


def get_num_true(outputs, inputs, has_ghost=True):
    """ get how many tracks should be included """
    from_decay = get_tracks_from_decay(inputs)
    return np.sum(from_decay, axis=1)


def get_num_false_negative(outputs, inputs, threshold=0.5, has_ghost=True):
    """ get count of tracks falsely identified by model as from decay """
    from_decay = get_tracks_from_decay(inputs)
    tracks_above_threshold = get_tracks_above_threshold(outputs, threshold, has_ghost=has_ghost)

    tracks_not_selected = np.logical_not(tracks_above_threshold)
    tracks_missed = np.logical_and(tracks_not_selected, from_decay)
    return np.sum(tracks_missed, axis=1)


def get_b_pred(outputs):
    """ return where predicted flavor is b (pb > pc,pu) """
    b_max = np.where(
        np.logical_and(
            outputs[:, daf.JetPrediction.PROB_B] > outputs[:, daf.JetPrediction.PROB_C],
            outputs[:, daf.JetPrediction.PROB_B] > outputs[:, daf.JetPrediction.PROB_U],
        ),
        1,
        0,
    )
    return b_max


def get_c_pred(outputs):
    """ return where predicted flavor is c (pc > pb,pu) """
    c_max = np.where(
        np.logical_and(
            outputs[:, daf.JetPrediction.PROB_C] > outputs[:, daf.JetPrediction.PROB_B],
            outputs[:, daf.JetPrediction.PROB_C] > outputs[:, daf.JetPrediction.PROB_U],
        ),
        1,
        0,
    )
    return c_max


def get_u_pred(outputs):
    """ return where predited flavor is u (pu > pb,pc) """
    u_max = np.where(
        np.logical_and(
            outputs[:, daf.JetPrediction.PROB_U] > outputs[:, daf.JetPrediction.PROB_C],
            outputs[:, daf.JetPrediction.PROB_U] > outputs[:, daf.JetPrediction.PROB_B],
        ),
        1,
        0,
    )
    return u_max
