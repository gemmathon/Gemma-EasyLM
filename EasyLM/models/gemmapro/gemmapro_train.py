import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from flax import traverse_util
from flax.core import freeze, frozen_dict
from flax.core.frozen_dict import FrozenDict
from typing import Mapping

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG,
    JaxDistributedConfig,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    global_norm,
    get_float_dtype_by_name,
    set_random_seed,
    average_metrics,
    get_weight_decay_mask,
    make_shard_and_gather_fns,
    with_sharding_constraint,
)
from EasyLM.models.gemmapro.gemmapro_model import FlaxGemmaForCausalLMModule
from EasyLM.models.gemmapro.configuration_gemmapro import GemmaProConfig

from transformers import AutoTokenizer

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim="1,-1,1",
    dtype="bf16",
    total_steps=10000,
    update_gemma_config="",
    load_checkpoint="",
    load_dataset_state="",
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    # tokenizer=GemmaConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    # gemma=GemmaConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    # tokenizer = GemmaConfig.get_tokenizer(FLAGS.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != "":
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    # if FLAGS.load_gemma_config != "":
    #     gemma_config = GemmaConfig.load_config(FLAGS.load_gemma_config)
    # else:
    #     gemma_config = GemmaConfig(**FLAGS.gemma)
    gemma_config = GemmaProConfig.from_pretrained("gemmathon/gemma-2b-pro")

    # if FLAGS.update_gemma_config != "":
    #     gemma_config.update(dict(eval(FLAGS.update_gemma_config)))

    # gemma_config.update(
    #     dict(
    #         bos_token_id=dataset.tokenizer.bos_token_id,
    #         eos_token_id=dataset.tokenizer.eos_token_id,
    #     )
    # )
    # if gemma_config.vocab_size < dataset.vocab_size:
    #     gemma_config.update(dict(vocab_size=dataset.vocab_size))

    model = FlaxGemmaForCausalLMModule(
        gemma_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GemmaProConfig.get_weight_decay_exclusions()),
    )


    def bool_grad_mask_fn(params):
        """Create a boolean mask over parameters for filtering.
        Returns:
            mask (frozen_dict): `True` for non-frozen parameters, `False` for frozen parameters (i.e. the feature encoder).
        """
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: ('20' in path or '6' in path or '13' in path) for path in flat_params}
        mask = traverse_util.unflatten_dict(flat_mask)
        return freeze(mask)
    
    def filter_params(params, mask):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = traverse_util.flatten_dict(mask)
        filtered_params = {}
        for name, value in flat_params.items():
            if flat_mask[name]:
                filtered_params[name] = flat_params[name]
        return freeze(traverse_util.unflatten_dict(filtered_params))

    def merge_params(params: Mapping, updates: Mapping) -> FrozenDict:
        if isinstance(params, FrozenDict):
            output = params.unfreeze()
        else:
            output = params

        for name, update_value in updates.items():
            current_value = params.get(name, None)
            if isinstance(current_value, Mapping) and isinstance(update_value, Mapping):
                output[name] = merge_params(current_value, update_value)
            else:
                output[name] = update_value

        return freeze(output)

    def create_trainstate_from_params(params):
        train_state = TrainState.create(params=params, tx=optimizer, apply_fn=None)
        # train_state = train_state.replace(
        #     opt_state = optimizer.init(filter_params(params, bool_grad_mask_fn(params)).unfreeze())
        # )
        return train_state

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(gemma_config.rng_keys()),
        )
        return create_trainstate_from_params(params)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        
        def loss_and_accuracy(params):
            # params = merge_params(params, differentiable_params)
            logits = model.apply(
                params,
                batch["input_tokens"],
                deterministic=False,
                rngs=rng_generator(gemma_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(
                logits, batch["target_tokens"], batch["loss_masks"]
            )
        
        # differentiable_params = filter_params(train_state.params, bool_grad_mask_fn(train_state.params))

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(train_state.params)
        # print(differentiable_params)
        # print(train_state.opt_state)

        new_state = train_state.apply_gradients(grads=grads)
        differentiable_params = filter_params(new_state.params, bool_grad_mask_fn(new_state.params))
        
        # print(differentiable_params)
        params = merge_params(train_state.params, differentiable_params).unfreeze()
        
        # updates, opt_state = train_state.tx.update(grads, train_state.opt_state, differentiable_params)
        
        # differentiable_params = optax.apply_updates(differentiable_params, updates)
        
        # params = merge_params(train_state.params, differentiable_params)
        new_state = new_state.replace(
            params=params,
        )

        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info["learning_rate_schedule"](new_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(new_state.params),
        )
        return new_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            train_state.params,
            batch["input_tokens"],
            deterministic=True,
            rngs=rng_generator(gemma_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch["target_tokens"], batch["loss_masks"]
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    # print("train_state_shapes:", train_state_shapes)
    train_state_partition = match_partition_rules(
        GemmaProConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn, in_shardings=PS(), out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            gemma_config=gemma_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = GemmaProConfig.get_jax_mesh(FLAGS.mesh_dim)
    print("Setup Mesh with:", mesh.shape, mesh.size)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != "":
            print("Loading checkpoint from", FLAGS.load_checkpoint)
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))
        print("Start training from step", start_step)

        # print("Test save_checkpoint")
        # if FLAGS.save_model_freq > 0:
        #     save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if (
                FLAGS.save_milestone_freq > 0
                and (step + 1) % FLAGS.save_milestone_freq == 0
            ):
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
