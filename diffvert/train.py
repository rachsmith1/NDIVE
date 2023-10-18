""" File used to train models.

Models trained can either be full flavor tagging models or a smaller model containing steps up to
and including differentiable vertexing layer.
"""
import jax
import jax.numpy as jnp
from jax.config import config

import flax
from flax.training import train_state, checkpoints
from flax.training.early_stopping import EarlyStopping
from flax.core.frozen_dict import freeze, unfreeze
import diffvert.models.train_config as tc
import optax

import numpy as np
import json

import datetime
import argparse
import importlib

import torch
import os

from functools import partial

import hashlib
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)


def parse_args():
    """ Argument parser for training script. """
    parser = argparse.ArgumentParser(description="Train the model.")

    parser.add_argument(
        "-m", "--model", default="empty", type=str, help="Which model to train."
    )

    parser.add_argument(
        "-s",
        "--samples",
        default="/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/all_flavors",
        type=str,
        help="Path to training samples.",
    )

    parser.add_argument(
        "-n", "--num_gpus", default=2, type=int, help="Number of GPUs to use."
    )

    parser.add_argument(
        "-b", "--batch_size", default=100, type=int, help="Number of jets per batch."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs for training.",
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate for training.",
    )

    parser.add_argument(
        "-p",
        "--pretrained_NDIVE",
        default=False,
        type=bool,
        help="Train the larger ftag model with pretrained NDIVE.",
    )

    parser.add_argument(
        "-c",
        "--cont",
        default=False,
        type=bool,
        help="Continue training a pretrained model.",
    )

    parser.add_argument(
        "-t",
        "--train_config",
        default="none",
        type=str,
        help="Specify training config.",
    )

    parser.add_argument(
        "--train_count",
        default=0,
        type=int,
        help="Specify count of identical model to train.",
    )

    args = parser.parse_args()
    return args


def persistent_hash(cfg: tc.TrainConfig) -> str:
    """ Get hash of config. 
    
    Meant to be random across configs, determined for a specific config.
    """
    sha = hashlib.sha256()
    sha.update(str(cfg).encode())
    return sha.hexdigest()[:6]


def get_ckpt_dir(
        cfg: tc.TrainConfig, epoch: int | None = None, model_number: int | None = None,
    ) -> str:
    """ return checkpoint directory given train config 
    
    Args:
        cfg: Training Config. Each config will be saved separately
        epoch: int or None specifying epoch of training. if none, will get over-written
        model_number: model iteration count (used for training multiple runs of model)
    Returns:
        directory of where model from cfg at epoch will be saved 
            if cfg has a name, it will be saved under that name. otherwise the cfg is hashed
    """
    prefix = os.getenv("NDIVE_MODEL_PATH")
    if cfg.config_name != "none":
        path = prefix + cfg.config_name
    else:
        path = prefix + f"{persistent_hash(cfg)}"
    if model_number is not None:
        path += f"/model_{model_number}"
    if epoch is not None:
        path += f"/{epoch}"
    return path


def get_most_recent_epoch(cfg: tc.TrainConfig, model_number: int|None=None) -> int:
    """ Find most recently saved epoch for a given config.

    Note that if you extend training by increasing num_epochs in the cfg it needs to be saved
        via config_name present in the cfg.
    """
    prefix_path = get_ckpt_dir(cfg, model_number=model_number)
    for epoch in range(cfg.num_epochs, 0, -1):
        if os.path.exists(f"{prefix_path}/{epoch}"): return epoch


def train_model(args, cfg: tc.TrainConfig, model_number=None):
    """ Run full training loop for a given train config.

    Saves checkpoints every few epochs, as well as most recent checkpoint.
    """
    print("starting train.py:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    train_dl = torch.load(f"{cfg.samples}/train_dl.pth")
    validate_dl = torch.load(f"{cfg.samples}/validate_dl.pth")

    print(
        "train and validate loaded:",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        flush=True,
    )

    model_from_config = getattr(
        importlib.import_module(f"diffvert.models.{cfg.model_name}"),
        "model_from_config",
    )
    loss_function_full = getattr(
        importlib.import_module(f"diffvert.models.{cfg.model_name}"),
        "loss_function",
    )
    loss_function = partial(loss_function_full, cfg=cfg)

    nominal_batch_size = 10000

    device_count = jax.device_count()
    train_vmap_count = int(nominal_batch_size / device_count / cfg.batch_size)
    test_vmap_count = int(nominal_batch_size / device_count / cfg.batch_size)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = model_from_config(cfg)

    print("starting model init:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    if args.cont:
        most_recent_epoch = get_most_recent_epoch(cfg, model_number)
        if most_recent_epoch is None:
            raise OSError("Previous checkpoint of model not found. Cannot use continuation.")
        params = checkpoints.restore_checkpoint(
            ckpt_dir=get_ckpt_dir(cfg, most_recent_epoch, model_number),
            target=None,
            step=0,
            parallel=False
        )["params"]
    else:
        params = model.init(init_rng, jnp.ones([10,15,51])*5, rng)["params"]
        if cfg.pretrained_NDIVE:
            params = unfreeze(params)
            # assumes specific saved location for trained ndive only
            ndive_params = checkpoints.restore_checkpoint(
                ckpt_dir=f"{os.getenv('NDIVE_MODEL_PATH')}ndive_only", target=None,
                step=0, parallel=False,
            )["model"]["params"]
            params["NDIVE"] = ndive_params
            params = freeze(params)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"number of parameters: {param_count}")
    print("done with model init:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          flush=True)

    total_steps = cfg.num_epochs*len(train_dl)*device_count*train_vmap_count
    cosine_decay_schedule = optax.cosine_decay_schedule(
       cfg.learning_rate, decay_steps=total_steps,# alpha=0.5,
    )

    learning_rate = cfg.learning_rate
    if cfg.use_cosine_decay_schedule:
        learning_rate = cosine_decay_schedule

    optimizer = optax.novograd(learning_rate=learning_rate)
    if cfg.use_adam:
        optimizer = optax.adam(learning_rate=learning_rate)

    tx=optimizer
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, 0), out_axes=(0, 0, 0))
    def train_step_pmap(key, state, batch_x, batch_y):
        def train_step(state, batch_x, batch_y):
            def loss_fn(params):
                outputs = state.apply_fn({"params": params}, batch_x, key)
                loss, losses = loss_function(batch_y, batch_x, outputs)
                return loss, losses

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, losses), grads = grad_fn(state.params)
            return loss, losses, grads

        train_step_vmap = jax.vmap(train_step, in_axes=(None, 0, 0), out_axes=(0, 0, 0))

        loss, losses, grads = train_step_vmap(state, batch_x, batch_y)
        loss_total = jnp.mean(loss)

        return loss_total, losses, grads

    @partial(jax.pmap, axis_name="device", in_axes=(None, 0, 0, 0), out_axes=(0, 0))
    def eval_step_pmap(key, state, batch_x, batch_y):
        def eval_step(state, batch_x, batch_y):
            outputs = state.apply_fn({"params": state.params}, batch_x, key)
            loss, losses = loss_function(batch_y, batch_x, outputs)
            return loss, losses

        eval_step_vmap = jax.vmap(eval_step, in_axes=(None, 0, 0), out_axes=(0, 0))

        loss, losses = eval_step_vmap(state, batch_x, batch_y)
        loss_total = jnp.mean(loss)

        return loss_total, losses

    @jax.jit
    def update_model(state, grads):
        def gradient_application(i, st):
            m = jnp.int32(jnp.floor(i / train_vmap_count))
            n = jnp.int32(jnp.mod(i, train_vmap_count))
            grad = jax.tree_util.tree_map(lambda x: x[m][n], grads)

            st = st.apply_gradients(grads=grad)
            return st

        state = jax.lax.fori_loop(
            0, device_count * train_vmap_count, gradient_application, state
        )

        return state

    def train_epoch(key, state, train_ds, epoch):
        state_dist = flax.jax_utils.replicate(state)

        batch_loss_total = []
        batch_losses = []

        for i, d in enumerate(train_ds):
            if i % 10 == 0:
                print(f"Batch #{i}", flush=True)
            # if i>=1: break

            x = jnp.array(d.x, dtype=jnp.float64)[:, :, 0:30]
            y = jnp.array(d.y, dtype=jnp.float64)

            # shuffle jets in each batch during training
            num_jets, _, _ = x.shape
            idx = jax.random.permutation(key, num_jets)
            x = x[idx]
            y = y[idx]

            x_batch = jax.tree_map(
                lambda m: m.reshape((device_count, train_vmap_count, -1, *m.shape[1:])),
                x,
            )
            y_batch = jax.tree_map(
                lambda m: m.reshape((device_count, train_vmap_count, -1, *m.shape[1:])),
                y,
            )

            # with jax.disable_jit():
            loss_total, losses, grads = train_step_pmap(
                key, state_dist, x_batch, y_batch
            )

            state = flax.jax_utils.unreplicate(state_dist)
            state = update_model(state, grads)
            state_dist = flax.jax_utils.replicate(state)

            batch_loss_total += list(loss_total)
            if len(batch_losses) == 0:
                for loss in losses:
                    batch_losses.append([np.mean(np.array(loss))])
            else:
                for i, loss in enumerate(losses):
                    batch_losses[i] += [np.mean(np.array(loss))]

        batch_loss_total = np.mean(batch_loss_total)
        batch_losses = np.array([np.mean(loss) for loss in batch_losses])

        print(f"Training - epoch: {epoch}, loss: {batch_loss_total}")
        for idx, loss in enumerate(batch_losses):
            print(f" - loss {idx}: {loss}")
        print("   finished at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return state, batch_loss_total, batch_losses

    def eval_model(key, state, test_ds, epoch):
        """ evaluate model over a dataset

        Args:
            key: PRNG key
            state: model state
            test_ds: batched data set
            epoch: epoch of training (used for printing)
        Returns:
            losses computed from evaluation
        """
        state_dist = flax.jax_utils.replicate(state)

        batch_loss_total = []
        batch_losses = []

        for i, d in enumerate(test_ds):
            if i % 10 == 0:
                print(f"Batch #{i}", flush=True)
            # if i>=1: break

            x = jnp.array(d.x, dtype=jnp.float64)[:, :, 0:30]
            y = jnp.array(d.y, dtype=jnp.float64)

            # shuffle jets in each batch during training
            num_jets = x.shape[0]
            idx = jax.random.permutation(key, num_jets)
            x = x[idx]
            y = y[idx]

            x_batch = jax.tree_map(
                lambda m: m.reshape((device_count, test_vmap_count, -1, *m.shape[1:])),
                x,
            )
            y_batch = jax.tree_map(
                lambda m: m.reshape((device_count, test_vmap_count, -1, *m.shape[1:])),
                y,
            )

            # with jax.disable_jit():
            loss_total, losses = eval_step_pmap(key, state_dist, x_batch, y_batch)

            batch_loss_total += list(loss_total)
            if len(batch_losses) == 0:
                for i, loss in enumerate(losses):
                    batch_losses.append([np.mean(np.array(loss))])
            else:
                for i, loss in enumerate(losses):
                    batch_losses[i] += [np.mean(np.array(loss))]

        batch_loss_total = np.mean(batch_loss_total)
        batch_losses = np.array([np.mean(loss) for loss in batch_losses])

        print(f"Validation - epoch: {epoch}, loss: {batch_loss_total}")
        for i, loss in enumerate(batch_losses):
            print(f" - loss {i}: {loss}")
        print("   finished at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return batch_loss_total, batch_losses

    start_epoch = 0
    if args.cont:
        data = []

        with open(get_ckpt_dir(cfg, most_recent_epoch, model_number)+"/losses.json", "r") as f:
            for line in f:
                data.append(json.loads(line))

        len_aux = int((len(data)-3)/2)

        start_epoch = data[0]
        train_total_losses = data[1]
        test_total_losses = data[2]
        train_aux_losses = data[3:3+len_aux]
        test_aux_losses = data[3+len_aux:3+2*len_aux]
    else:
        train_total_losses = []
        test_total_losses = []

        train_aux_losses = []
        test_aux_losses = []

    print("starting epochs:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("save directory:", get_ckpt_dir(cfg, model_number=model_number))

    early_stop = EarlyStopping(min_delta=1e-6, patience=20)
    save_every_epoch_count = 10

    max_epochs = cfg.num_epochs
    if cfg.use_early_stopping: max_epochs = 1000 # make effectively infinit
    stalled_epochs = 0

    for epoch in range(1, max_epochs + 1):
        key = jax.random.PRNGKey(datetime.datetime.now().second)

        state, train_loss_total, train_losses = train_epoch(key, state, train_dl, epoch)
        test_loss_total, test_losses = eval_model(key, state, validate_dl, epoch)

        train_total_losses.append(float(train_loss_total))
        test_total_losses.append(float(test_loss_total))

        has_improved, early_stop = early_stop.update(test_loss_total)

        if len(train_aux_losses) == 0:
            for loss in train_losses:
                train_aux_losses.append(list([float(loss)]))
            for loss in test_losses:
                test_aux_losses.append(list([float(loss)]))
        else:
            for i, loss in enumerate(train_losses):
                train_aux_losses[i].append(float(loss))
            for i, loss in enumerate(test_losses):
                test_aux_losses[i].append(float(loss))

        if epoch >= 1:
            ckpt = {"model": state, "config": cfg}
            checkpoints.save_checkpoint(
                ckpt_dir=get_ckpt_dir(cfg, model_number=model_number),
                target=ckpt,
                step=0,
                overwrite=True,
            )
            if epoch % save_every_epoch_count == 0:
                checkpoints.save_checkpoint(
                    ckpt_dir=get_ckpt_dir(cfg, epoch, model_number),
                    target=ckpt,
                    step=0,
                    overwrite=True,
                )

            loss_files = [get_ckpt_dir(cfg, model_number=model_number) + "/losses.json"]
            if epoch % save_every_epoch_count == 0:
                loss_files.append(get_ckpt_dir(cfg,epoch, model_number) + "/losses.json")
            for loss_file in loss_files:
                with open(loss_file, "w") as f:
                    f.write(json.dumps(epoch + start_epoch))
                    f.write("\n")
                    f.write(json.dumps(train_total_losses))
                    f.write("\n")
                    f.write(json.dumps(test_total_losses))
                    for i, loss in enumerate(train_aux_losses):
                        f.write("\n")
                        f.write(json.dumps(loss))
                    for i, loss in enumerate(test_aux_losses):
                        f.write("\n")
                        f.write(json.dumps(loss))

        if cfg.use_early_stopping and early_stop.should_stop:
            print(f"Early stopping on epoch {epoch}")
            break

        if not has_improved:
            stalled_epochs+=1
            # lower learning rate if see 5 epochs without improvement
            if stalled_epochs == 5 and cfg.use_learning_rate_decay_when_stalled:
                print(f"decreasing learning rate from {learning_rate} to {learning_rate/10}")
                learning_rate = learning_rate/10
                optimizer = optax.novograd(learning_rate=learning_rate)
                if cfg.use_adam:
                    optimizer = optax.adam(learning_rate=learning_rate)
                state = train_state.TrainState.create(
                    apply_fn=model.apply, params=state.params, tx=optimizer,
                )
                stalled_epochs = 0
        else:
            stalled_epochs = 0


def main():
    args = parse_args()

    cfg = tc.TrainConfig( # edit this to change training
        model_name = "ftag",
        num_epochs = 200,
        samples = "/gpfs/slac/atlas/fs1/d/recsmith/Vertexing/samples/all_flavors/all_flavors",
        batch_size = 1000,
        learning_rate = 1e-4,
        pretrained_NDIVE = False,
        track_weight_activation = int(tc.WeightActivation.SIGMOID),
        num_attention_layers=3,
        num_attention_heads=2,
        jet_flavor_loss=True,
        track_origin_loss=True,
        track_pairing_loss=True,
        vertex_loss=False,
        use_mse_loss=False,
        normalize_vertex_loss=False,
        chi_squared_loss=False,
        track_weight_loss=False,
        vertexer=int(tc.Vertexer.NONE),
        use_ghost_track=False,
        clip_vertex=True,
        use_one_hot_encoding=False,
        use_early_stopping=True,
        use_adam=True,
        use_learning_rate_decay_when_stalled=True,
        config_name="test_three",
    )

    if args.train_config != "none":
        cfg = getattr(
            importlib.import_module("diffvert.configs"),
            args.train_config,
        )

    print("config:", cfg)
    if args.train_count > 0:
        
        new_idx = 0
        model_number_in_directory = True
        while model_number_in_directory:
            prefix_path = get_ckpt_dir(cfg, model_number=new_idx)
            if os.path.exists(f"{prefix_path}"): 
                new_idx += 1
            else:
                model_number_in_directory = False
                
        for idx in range(args.train_count):
            print(f"Training model number {idx + new_idx}.")
            train_model(args, cfg, model_number=idx + new_idx)
    else: # use train_count 0 as standin for not using multiple trainings
        train_model(args, cfg)


if __name__ == "__main__":
    main()
