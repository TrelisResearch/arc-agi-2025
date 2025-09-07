import math
import os
import time
from functools import partial
from typing import Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from matplotlib import pyplot as plt

# Assuming 'llm_python' is a local module you have.
# Replace this with your actual ARC data loading utility.
from llm_python.utils.task_loader import get_task_loader

# --- 1. Constants and Configuration ---

# Define palette and special values for the grid
MASK_VALUE: int = 10
VOCAB_SIZE: int = 11  # 0-9 for colors, 10 for mask
MAX_GRID_DIM: int = 30  # Pad all grids to 30x30

# --- 2. U-Net Architecture (Time-Aware with Dropout) ---


class SinusoidalPosEmb(eqx.Module):
    """Standard sinusoidal time embedding module."""

    dim: int

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        # This module expects a 1D array of timesteps (e.g., shape (B,) or (1,))
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freq = jnp.exp(jnp.arange(half_dim) * -scale)
        args = time[:, None] * freq[None, :]
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return embedding


class TimeEmbeddingBlock(eqx.Module):
    """Projects the time embedding to the right dimension."""

    mlp: List[eqx.nn.Linear]

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.Array):
        keys = jax.random.split(key, 2)
        self.mlp = [
            eqx.nn.Linear(in_channels, out_channels, key=keys[0]),
            eqx.nn.Linear(out_channels, out_channels, key=keys[1]),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.silu(self.mlp[0](x))
        return self.mlp[1](x)


class ColorPermutationGenerator(eqx.Module):
    """
    A hypernetwork that generates a color permutation matrix based on the source grid content and task ID.
    """

    cnn: eqx.nn.Sequential
    mlp: eqx.nn.MLP
    output_size: int = eqx.field(static=True)
    num_tasks: int = eqx.field(static=True)

    def __init__(self, num_tasks: int, vocab_size: int, *, key: jax.Array):
        cnn_key1, cnn_key2, mlp_key = jax.random.split(key, 3)
        self.output_size = vocab_size * vocab_size
        self.num_tasks = num_tasks

        # A simple CNN to process the grid and extract features
        self.cnn = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    vocab_size, 64, kernel_size=3, stride=1, padding=1, key=cnn_key1
                ),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv2d(
                    64, 128, kernel_size=3, stride=2, padding=1, key=cnn_key2
                ),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.AvgPool2d(kernel_size=2, stride=2),
                eqx.nn.Lambda(jnp.ravel),  # Flatten the features
            ]
        )

        # Dummy forward pass to calculate flattened size
        dummy_input = jnp.zeros((vocab_size, MAX_GRID_DIM, MAX_GRID_DIM))
        cnn_output_size = self.cnn(dummy_input).shape[0]

        # MLP input size now includes the one-hot task vector
        self.mlp = eqx.nn.MLP(
            in_size=cnn_output_size + self.num_tasks,
            out_size=self.output_size,
            width_size=128,
            depth=2,
            key=mlp_key,
        )

    def __call__(
        self, source_grid: jnp.ndarray, task_id: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Permutation is based on the source grid and task_id
        source_onehot = jax.nn.one_hot(source_grid, VOCAB_SIZE)
        source_ch_first = jnp.transpose(source_onehot, (2, 0, 1))

        grid_features = self.cnn(source_ch_first)

        # Create one-hot vector for the task
        task_onehot = jax.nn.one_hot(task_id, num_classes=self.num_tasks)

        # Concatenate grid features and task vector
        combined_features = jnp.concatenate([grid_features, task_onehot])

        flat_matrix = self.mlp(combined_features)
        matrix = flat_matrix.reshape((VOCAB_SIZE, VOCAB_SIZE))

        soft_permutation_matrix = jax.nn.softmax(matrix, axis=0)
        inverse_soft_permutation_matrix = soft_permutation_matrix.T

        return soft_permutation_matrix, inverse_soft_permutation_matrix


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    time_mlp: TimeEmbeddingBlock
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout_p: float = 0.2,
        *,
        key: jax.Array,
    ):
        conv_key, time_key = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=conv_key
        )
        self.norm = eqx.nn.GroupNorm(groups=8, channels=out_channels)
        self.time_mlp = TimeEmbeddingBlock(time_embed_dim, out_channels, key=time_key)
        self.dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(
        self, x: jnp.ndarray, time_emb: jnp.ndarray, *, key: jax.Array
    ) -> jnp.ndarray:
        # x is (C, H, W), time_emb is (D,)
        x = self.conv(x)
        x = self.norm(x)

        # Add timestep conditioning
        time_cond = self.time_mlp(time_emb)
        x = x + time_cond[:, None, None]  # Broadcast to grid size

        x = jax.nn.silu(x)
        x = self.dropout(x, key=key)
        return x


class ArcUNetSolver(eqx.Module):
    num_tasks: int = eqx.field(static=True)
    time_embed_dim: int = eqx.field(static=True)

    # The hypernetwork for generating color permutations
    permutation_generator: ColorPermutationGenerator

    time_embedding_projector: SinusoidalPosEmb
    time_mlp: TimeEmbeddingBlock
    down_block1: List[ConvBlock]
    down_block2: List[ConvBlock]
    mid_block1: ConvBlock
    mid_block2: ConvBlock
    up_block1: List[ConvBlock]
    up_block2: List[ConvBlock]
    final_conv: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d

    def __init__(
        self,
        num_tasks: int,
        base_channels: int = 256,
        time_embed_dim: int = 32,
        dropout_p: float = 0.2,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 11)
        self.num_tasks = num_tasks
        self.time_embed_dim = time_embed_dim

        self.permutation_generator = ColorPermutationGenerator(
            num_tasks, VOCAB_SIZE, key=keys[0]
        )
        self.time_embedding_projector = SinusoidalPosEmb(dim=time_embed_dim)
        self.time_mlp = TimeEmbeddingBlock(time_embed_dim, time_embed_dim, key=keys[1])

        # The input to the U-Net is one-hot again after permutation
        in_ch = 2 * VOCAB_SIZE + self.num_tasks
        c1, c2 = base_channels, base_channels * 2

        self.down_block1 = [
            ConvBlock(in_ch, c1, time_embed_dim, dropout_p, key=keys[2]),
            ConvBlock(c1, c1, time_embed_dim, dropout_p, key=keys[3]),
        ]
        self.down_block2 = [
            ConvBlock(c1, c2, time_embed_dim, dropout_p, key=keys[4]),
            ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[5]),
        ]
        self.mid_block1 = ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[6])
        self.mid_block2 = ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[7])
        self.up_block1 = [
            ConvBlock(c2 + c2, c1, time_embed_dim, dropout_p, key=keys[8]),
            ConvBlock(c1, c1, time_embed_dim, dropout_p, key=keys[9]),
        ]
        self.up_block2 = [
            ConvBlock(c1 + c1, c1, time_embed_dim, dropout_p, key=keys[10])
        ]
        self.final_conv = eqx.nn.Conv2d(c1, VOCAB_SIZE, kernel_size=1, key=keys[10])
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(
        self,
        source_grid: jnp.ndarray,
        masked_target_grid: jnp.ndarray,
        task_id: jnp.ndarray,
        timestep: jnp.ndarray,
        *,
        key: jax.Array,
    ) -> jnp.ndarray:
        if key is not None:
            keys = jax.random.split(key, 9)
        else:
            keys = [None] * 9

        # 1. Generate the content- and task-aware permutation matrices
        P, P_inv = self.permutation_generator(source_grid, task_id)

        # 2. One-hot encode and then permute the input grids
        source_onehot = jax.nn.one_hot(source_grid, VOCAB_SIZE)
        permuted_source = source_onehot @ P

        masked_target_onehot = jax.nn.one_hot(masked_target_grid, VOCAB_SIZE)
        permuted_masked_target = masked_target_onehot @ P

        # Transpose to (C, H, W) for convolutions
        permuted_source_ch = jnp.transpose(permuted_source, (2, 0, 1))
        permuted_masked_target_ch = jnp.transpose(permuted_masked_target, (2, 0, 1))

        task_onehot = jax.nn.one_hot(task_id, num_classes=self.num_tasks)
        task_cond = jnp.broadcast_to(
            task_onehot[:, None, None], (self.num_tasks, MAX_GRID_DIM, MAX_GRID_DIM)
        )
        x = jnp.concatenate(
            [permuted_source_ch, permuted_masked_target_ch, task_cond], axis=0
        )

        # Timestep embedding for a single scalar
        time_emb_batch = self.time_embedding_projector(timestep[None])
        time_emb = self.time_mlp(time_emb_batch[0])

        # U-Net operates in the permuted color space
        skip1 = self.down_block1[1](
            self.down_block1[0](x, time_emb, key=keys[0]), time_emb, key=keys[1]
        )
        x = self.pool(skip1)
        skip2 = self.down_block2[1](
            self.down_block2[0](x, time_emb, key=keys[2]), time_emb, key=keys[3]
        )
        x = self.pool(skip2)
        x = self.mid_block1(x, time_emb, key=keys[4])
        x = self.mid_block2(x, time_emb, key=keys[5])
        x = jax.image.resize(x, (x.shape[0],) + skip2.shape[1:], "nearest")
        x = jnp.concatenate([x, skip2], axis=0)
        x = self.up_block1[1](
            self.up_block1[0](x, time_emb, key=keys[6]), time_emb, key=keys[7]
        )
        x = jax.image.resize(x, (x.shape[0],) + skip1.shape[1:], "nearest")
        x = jnp.concatenate([x, skip1], axis=0)
        x = self.up_block2[0](x, time_emb, key=keys[8])

        permuted_logits_ch = self.final_conv(x)
        permuted_logits = jnp.transpose(permuted_logits_ch, (1, 2, 0))

        # 3. Apply the inverse permutation to the output logits
        final_logits = permuted_logits @ P_inv

        return final_logits


# --- 3. Data Pipeline ---


def pad_grid(grid: np.ndarray, max_dim: int = MAX_GRID_DIM) -> np.ndarray:
    h, w = grid.shape
    padded_grid = np.full((max_dim, max_dim), 0, dtype=np.int32)
    padded_grid[:h, :w] = grid
    return padded_grid


# --- 4. Training and Evaluation Logic ---


def loss_fn(
    model: ArcUNetSolver,
    source_grid: jnp.ndarray,
    final_target_grid: jnp.ndarray,
    task_id: jnp.ndarray,
    corrupted_target: jnp.ndarray,
    loss_mask: jnp.ndarray,
    timestep: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    pred_logits = model(source_grid, corrupted_target, task_id, timestep, key=key)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        pred_logits, final_target_grid
    )
    masked_loss = (losses * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1)
    return masked_loss


def train_step(
    model: ArcUNetSolver,
    optim_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    key: jax.Array,
    noise_ratio: float,
) -> Tuple[ArcUNetSolver, optax.OptState, jnp.ndarray]:
    
    def compute_batch_loss(model, batch, key):
        # Create separate keys for each random operation
        ratio_key, key = jax.random.split(key)
        keys_for_vmap = jax.random.split(key, 4)
        masking_keys, noise_keys, color_keys, loss_keys = [
            jax.random.split(k, batch["source"].shape[0]) for k in keys_for_vmap
        ]

        # Get a random mask ratio for each example in the batch
        random_ratios = jax.random.uniform(ratio_key, shape=(batch["source"].shape[0],))

        # vmap the corruption logic over the batch
        def create_corruption_and_mask(target, ratio, mkey, nkey, ckey):
            can_be_masked = (target >= 0) & (target < 10)
            noise_for_mask = jax.random.uniform(mkey, shape=target.shape)
            mask_to_hide = (noise_for_mask < ratio) & can_be_masked
            can_be_noised = ~mask_to_hide & can_be_masked
            noise_for_noise = jax.random.uniform(nkey, shape=target.shape)
            mask_to_noise = (noise_for_noise < noise_ratio) & can_be_noised
            random_colors = jax.random.randint(
                ckey, shape=target.shape, minval=0, maxval=9
            )
            noised_target = jnp.where(mask_to_noise, random_colors, target)
            corrupted_target = jnp.where(mask_to_hide, MASK_VALUE, noised_target)
            loss_mask = mask_to_hide | mask_to_noise
            return corrupted_target, loss_mask

        corrupted_targets, loss_masks = jax.vmap(create_corruption_and_mask)(
            batch["target"], random_ratios, masking_keys, noise_keys, color_keys
        )

        loss_fn_partial = partial(loss_fn, model)
        vmapped_loss_fn = jax.vmap(loss_fn_partial)
        per_example_losses = vmapped_loss_fn(
            batch["source"],
            batch["target"],
            batch["task_id"],
            corrupted_targets,
            loss_masks,
            random_ratios,
            loss_keys,
        )
        return jnp.mean(per_example_losses)

    loss_val, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch, key)
    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optim_state, loss_val


def generate_with_diffusion(
    model: ArcUNetSolver,
    source_grid: jnp.ndarray,
    task_id: jnp.ndarray,
    key: jax.Array,
    num_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates a solution grid using an iterative diffusion process.
    Always returns the final grid and the full trace of intermediate steps.
    """

    def diffusion_step(i, state):
        current_target, trace_so_far, key = state
        trace_so_far = trace_so_far.at[i].set(current_target)

        key, model_key, sample_key, noise_key = jax.random.split(key, 4)

        power = 3
        timestep = jnp.array(((num_steps - 1 - i) / num_steps) ** power)

        pred_logits = model(
            source_grid, current_target, task_id, timestep, key=model_key
        )

        sampled_grid = jax.random.categorical(sample_key, pred_logits)
        
        update_ratio = (1.0 / (num_steps - i + 1)) * 2
        noise = jax.random.uniform(noise_key, shape=source_grid.shape)
        
        should_update_mask = (noise < update_ratio) & (current_target == MASK_VALUE)
        
        new_target = jnp.where(should_update_mask, sampled_grid, current_target)

        return new_target, trace_so_far, key

    initial_target = jnp.full_like(source_grid, MASK_VALUE)
    initial_trace = jnp.zeros((num_steps,) + source_grid.shape, dtype=jnp.int32)
    initial_state = (initial_target, initial_trace, key)

    final_target, final_trace, _ = jax.lax.fori_loop(
        0, num_steps, diffusion_step, initial_state
    )
    return final_target, final_trace


def visualize_diffusion_process(
    source_grid, target_grid, final_target, final_trace, save_path
):
    """
    Saves a grid of visualizations for a pre-computed diffusion process.
    """
    # --- Plotting Logic ---
    cmap = plt.cm.colors.ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
            "#D3D3D3",
        ]
    )
    norm = plt.cm.colors.Normalize(vmin=0, vmax=10)

    # Convert JAX arrays to NumPy arrays for plotting
    source_grid_np = np.asarray(source_grid)
    final_trace_np = np.asarray(final_trace)
    final_target_np = np.asarray(final_target)
    target_grid_np = np.asarray(target_grid)

    # Select which steps to show
    plots_to_show = [("Source", source_grid_np)]
    num_trace_steps = len(final_trace_np)
    max_intermediate_plots = 6

    if num_trace_steps <= max_intermediate_plots:
        indices_to_show = range(num_trace_steps)
    else:
        indices_to_show = np.linspace(
            0, num_trace_steps - 1, max_intermediate_plots, dtype=int
        )

    for i in indices_to_show:
        plots_to_show.append((f"Step {i + 1}", final_trace_np[i]))

    plots_to_show.append(("Final Pred.", final_target_np))
    plots_to_show.append(("Ground Truth", target_grid_np))

    num_plots = len(plots_to_show)
    cols = min(8, num_plots)
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (title, grid_data) in enumerate(plots_to_show):
        ax = axes[i]
        ax.imshow(grid_data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    for i in range(num_plots, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def evaluate_generation_accuracy(
    model: ArcUNetSolver,
    dataset: Dict[str, jnp.ndarray],
    key: jax.Array,
    num_diffusion_steps: int,
    batch_size: int = 32,
) -> Tuple[float, float]:
    model = eqx.tree_inference(model, value=True)
    perfect_count = 0
    total_correct_pixels = 0
    total_pixels = 0
    total_count = dataset["source"].shape[0]
    if total_count == 0:
        return 0.0, 0.0
    
    generate_partial = partial(generate_with_diffusion, model, num_steps=num_diffusion_steps)
    vmapped_generate = eqx.filter_jit(jax.vmap(generate_partial))

    for i in range(0, total_count, batch_size):
        start_index = i
        end_index = i + batch_size
        batch = jax.tree_util.tree_map(lambda x: x[start_index:end_index], dataset)

        key, step_key = jax.random.split(key)
        batch_keys = jax.random.split(step_key, batch["source"].shape[0])

        generated_grids, _ = vmapped_generate(
            batch["source"], batch["task_id"], batch_keys
        )
        matches = generated_grids == batch["target"]

        perfect_count += int(jnp.all(matches, axis=(1, 2)).sum().item())
        total_correct_pixels += matches.sum()
        total_pixels += batch["target"].size

    percent_perfect = 100.0 * perfect_count / total_count
    pixel_accuracy = 100.0 * total_correct_pixels / total_pixels
    return percent_perfect, pixel_accuracy


# --- 5. Main function to run the experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_TASKS = 400
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 2000
    NUM_DIFFUSION_STEPS = 1
    DROPOUT_RATE = 0.1
    BASE_CHANNELS = 256
    NOISE_RATIO = 0.2

    # --- Data Loading ---
    print("Loading and processing ARC data into grids...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")
    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}
    train_data_list = []
    eval_data_list = []
    # Create a reverse map for visualization
    number_to_task_id = {i: task_id for task_id, i in task_id_to_number.items()}

    for task_id, task in training_tasks[:NUM_TASKS]:
        task_num = task_id_to_number[task_id]
        for pair in task["train"]:
            if pair["input"] is None or pair["output"] is None:
                continue
            train_data_list.append(
                {
                    "source": pad_grid(np.array(pair["input"])),
                    "target": pad_grid(np.array(pair["output"])),
                    "task_id": task_num,
                }
            )
        for pair in task["test"]:
            if pair["input"] is None or pair["output"] is None:
                continue
            eval_data_list.append(
                {
                    "source": pad_grid(np.array(pair["input"])),
                    "target": pad_grid(np.array(pair["output"])),
                    "task_id": task_num,
                }
            )

    train_dataset_cpu = {
        "source": np.array([d["source"] for d in train_data_list]),
        "target": np.array([d["target"] for d in train_data_list]),
        "task_id": np.array([d["task_id"] for d in train_data_list]),
    }
    eval_dataset_cpu = {
        "source": np.array([d["source"] for d in eval_data_list]),
        "target": np.array([d["target"] for d in eval_data_list]),
        "task_id": np.array([d["task_id"] for d in eval_data_list]),
    }
    print(
        f"Loaded {train_dataset_cpu['source'].shape[0]} training pairs and {eval_dataset_cpu['source'].shape[0]} evaluation pairs."
    )

    # --- Model Initialization ---
    key = jax.random.PRNGKey(int(time.time()))
    model_key, train_key, eval_key, loader_key = jax.random.split(key, 4)
    model = ArcUNetSolver(
        num_tasks=NUM_TASKS,
        base_channels=BASE_CHANNELS,
        dropout_p=DROPOUT_RATE,
        key=model_key,
    )
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def count_params(model: eqx.Module):
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )

    print(f"Model has {count_params(model):,} trainable parameters.")

    print(f"Training U-Net for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}...")
    num_train_samples = train_dataset_cpu["source"].shape[0]
    time_since_eval = time.time()

    jitted_train_step = eqx.filter_jit(partial(train_step, noise_ratio=NOISE_RATIO))

    for epoch in range(NUM_EPOCHS):
        loader_key, perm_key = jax.random.split(loader_key)
        indices = jax.random.permutation(perm_key, num_train_samples)

        shuffled_train_dataset_cpu = jax.tree_util.tree_map(
            lambda x: x[indices], train_dataset_cpu
        )

        aggregate_train_loss = 0
        for i in range(0, num_train_samples, BATCH_SIZE):
            start_index = i
            end_index = i + BATCH_SIZE
            batch_cpu = jax.tree_util.tree_map(
                lambda x: x[start_index:end_index], shuffled_train_dataset_cpu
            )
            if batch_cpu["source"].shape[0] != BATCH_SIZE:
                continue

            batch_gpu = jax.tree_util.tree_map(jnp.array, batch_cpu)

            train_key, step_key = jax.random.split(train_key)
            model, optim_state, train_loss = jitted_train_step(
                model, optim_state, batch_gpu, step_key
            )
            aggregate_train_loss += train_loss.sum().item()

        if time.time() - time_since_eval > 60 or epoch == NUM_EPOCHS - 1:
            time_since_eval = time.time()
            eval_key, train_loss_key, train_acc_key, eval_acc_key = jax.random.split(
                eval_key, 4
            )

            model_for_eval = eqx.tree_inference(model, value=True)

            train_dataset_gpu = jax.tree_util.tree_map(jnp.array, train_dataset_cpu)
            eval_dataset_gpu = jax.tree_util.tree_map(jnp.array, eval_dataset_cpu)

            train_acc, train_pixel_acc = evaluate_generation_accuracy(
                model_for_eval,
                train_dataset_gpu,
                train_acc_key,
                NUM_DIFFUSION_STEPS,
                batch_size=BATCH_SIZE,
            )
            eval_acc, eval_pixel_acc = evaluate_generation_accuracy(
                model_for_eval,
                eval_dataset_gpu,
                eval_acc_key,
                NUM_DIFFUSION_STEPS,
                batch_size=BATCH_SIZE,
            )

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | TL: {aggregate_train_loss:.4f} | TA: {train_acc:.2f}% ({train_pixel_acc:.2f}%) | EA: {eval_acc:.2f}% ({eval_pixel_acc:.2f}%)"
            )

    # Final evaluation with the final model
    print("\n--- Final evaluation and visualization ---")
    model_for_eval = eqx.tree_inference(model, value=True)
    eval_dataset_gpu = jax.tree_util.tree_map(jnp.array, eval_dataset_cpu)

    final_acc, final_pixel_acc = evaluate_generation_accuracy(
        model_for_eval,
        eval_dataset_gpu,
        jax.random.PRNGKey(42),
        NUM_DIFFUSION_STEPS,
        batch_size=BATCH_SIZE,
    )
    print(
        f"Final Eval Perfect Solutions: {final_acc:.2f}% (Pixel Accuracy: {final_pixel_acc:.2f}%)"
    )

    # Visualize a few random examples from the eval set
    vis_key = jax.random.PRNGKey(43)
    num_visualizations = 10
    vis_indices = jax.random.choice(
        vis_key,
        eval_dataset_gpu["source"].shape[0],
        shape=(num_visualizations,),
        replace=False,
    )

    if not os.path.exists("diffusion_traces"):
        os.makedirs("diffusion_traces")

    # JIT the single-example generation function for visualization
    jitted_generate_single = eqx.filter_jit(generate_with_diffusion)

    for i in range(num_visualizations):
        idx = vis_indices[i]
        source = eval_dataset_gpu["source"][idx]
        target = eval_dataset_gpu["target"][idx]
        task_id_num = eval_dataset_gpu["task_id"][idx]
        task_id_str = number_to_task_id[int(task_id_num)].replace("/", "_")

        vis_key, step_key = jax.random.split(vis_key)

        # Generate the data needed for plotting
        final_target, final_trace = jitted_generate_single(
            model_for_eval, source, task_id_num, step_key, num_steps=NUM_DIFFUSION_STEPS
        )

        save_path = f"diffusion_traces/trace_example_{task_id_str}.png"
        # Pass the pre-computed data to the plotting function
        visualize_diffusion_process(
            source,
            target,
            final_target,
            final_trace,
            save_path=save_path,
        )
