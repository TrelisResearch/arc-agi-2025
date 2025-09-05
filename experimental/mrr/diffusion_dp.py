import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List, Tuple
from functools import partial
import math

# Assuming 'llm_python' is a local module you have.
# Replace this with your actual ARC data loading utility.
from llm_python.utils.task_loader import get_task_loader
import numpy as np

# --- 1. Constants and Configuration ---

# Define palette and special values for the grid
MASK_VALUE = 10
VOCAB_SIZE = 11  # 0-9 for colors, 10 for mask
MAX_GRID_DIM = 30 # Pad all grids to 30x30

# --- 2. U-Net Architecture (Time-Aware with Dropout) ---

class SinusoidalPosEmb(eqx.Module):
    """Standard sinusoidal time embedding module."""
    dim: int

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        # This module expects a 1D array of timesteps (e.g., shape (B,) or (1,))
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings

class TimeEmbeddingBlock(eqx.Module):
    """Projects the time embedding to the right dimension."""
    mlp: List[eqx.nn.Linear]

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.Array):
        keys = jax.random.split(key, 2)
        self.mlp = [
            eqx.nn.Linear(in_channels, out_channels, key=keys[0]),
            eqx.nn.Linear(out_channels, out_channels, key=keys[1])
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.silu(self.mlp[0](x))
        return self.mlp[1](x)

class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    time_mlp: TimeEmbeddingBlock
    dropout: eqx.nn.Dropout
    
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout_p=0.2, *, key: jax.Array):
        conv_key, time_key = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=conv_key)
        self.norm = eqx.nn.GroupNorm(groups=8, channels=out_channels)
        self.time_mlp = TimeEmbeddingBlock(time_embed_dim, out_channels, key=time_key)
        self.dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x: jnp.ndarray, time_emb: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        # x is (C, H, W), time_emb is (D,)
        x_res = x
        x = self.conv(x)
        x = self.norm(x)
        
        time_cond = self.time_mlp(time_emb)
        x = x + time_cond[:, None, None] # Broadcast to grid size
        
        x = jax.nn.silu(x)
        x = self.dropout(x, key=key)
        return x

class ArcUNetSolver(eqx.Module):
    num_tasks: int = eqx.field(static=True)
    time_embed_dim: int = eqx.field(static=True)
    
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
    
    def __init__(self, num_tasks: int, base_channels: int = 256, time_embed_dim: int = 32, dropout_p: float = 0.2, *, key: jax.Array):
        keys = jax.random.split(key, 10)
        self.num_tasks = num_tasks
        self.time_embed_dim = time_embed_dim

        self.time_embedding_projector = SinusoidalPosEmb(dim=time_embed_dim)
        self.time_mlp = TimeEmbeddingBlock(time_embed_dim, time_embed_dim, key=keys[0])

        in_ch = 2 * VOCAB_SIZE + self.num_tasks
        c1, c2 = base_channels, base_channels * 2
        
        self.down_block1 = [ConvBlock(in_ch, c1, time_embed_dim, dropout_p, key=keys[1]), ConvBlock(c1, c1, time_embed_dim, dropout_p, key=keys[2])]
        self.down_block2 = [ConvBlock(c1, c2, time_embed_dim, dropout_p, key=keys[3]), ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[4])]
        self.mid_block1 = ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[5])
        self.mid_block2 = ConvBlock(c2, c2, time_embed_dim, dropout_p, key=keys[6])
        self.up_block1 = [ConvBlock(c2 + c2, c1, time_embed_dim, dropout_p, key=keys[7]), ConvBlock(c1, c1, time_embed_dim, dropout_p, key=keys[8])]
        self.up_block2 = [ConvBlock(c1 + c1, c1, time_embed_dim, dropout_p, key=keys[9])]
        self.final_conv = eqx.nn.Conv2d(c1, VOCAB_SIZE, kernel_size=1, key=keys[9])
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, source_grid: jnp.ndarray, masked_target_grid: jnp.ndarray, task_id: jnp.ndarray, timestep: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        if key is not None:
            keys = jax.random.split(key, 9)
        else:
            keys = [None] * 9

        # Model is single-example aware: expects (H, W), not (B, H, W)
        source_onehot = jax.nn.one_hot(source_grid, VOCAB_SIZE)
        masked_target_onehot = jax.nn.one_hot(masked_target_grid, VOCAB_SIZE)
        source_onehot = jnp.transpose(source_onehot, (2, 0, 1))
        masked_target_onehot = jnp.transpose(masked_target_onehot, (2, 0, 1))
        
        task_onehot = jax.nn.one_hot(task_id, num_classes=self.num_tasks)
        task_cond = jnp.broadcast_to(task_onehot[:, None, None], (self.num_tasks, MAX_GRID_DIM, MAX_GRID_DIM))
        x = jnp.concatenate([source_onehot, masked_target_onehot, task_cond], axis=0)

        # Timestep embedding for a single scalar
        time_emb_batch = self.time_embedding_projector(timestep[None]) # Add batch dim -> (1, D)
        time_emb = self.time_mlp(time_emb_batch[0]) # Remove batch dim -> (D,)

        # U-Net with time conditioning
        skip1 = self.down_block1[1](self.down_block1[0](x, time_emb, key=keys[0]), time_emb, key=keys[1])
        x = self.pool(skip1)
        skip2 = self.down_block2[1](self.down_block2[0](x, time_emb, key=keys[2]), time_emb, key=keys[3])
        x = self.pool(skip2)
        x = self.mid_block1(x, time_emb, key=keys[4])
        x = self.mid_block2(x, time_emb, key=keys[5])
        x = jax.image.resize(x, (x.shape[0],) + skip2.shape[1:], "nearest")
        x = jnp.concatenate([x, skip2], axis=0)
        x = self.up_block1[1](self.up_block1[0](x, time_emb, key=keys[6]), time_emb, key=keys[7])
        x = jax.image.resize(x, (x.shape[0],) + skip1.shape[1:], "nearest")
        x = jnp.concatenate([x, skip1], axis=0)
        x = self.up_block2[0](x, time_emb, key=keys[8])
        
        logits = self.final_conv(x)
        return jnp.transpose(logits, (1, 2, 0))

# --- 3. Data Pipeline ---

def pad_grid(grid: np.ndarray, max_dim: int = MAX_GRID_DIM) -> np.ndarray:
    h, w = grid.shape
    padded_grid = np.full((max_dim, max_dim), 0, dtype=np.int32)
    padded_grid[:h, :w] = grid
    return padded_grid

# --- 4. Training and Evaluation Logic ---

def loss_fn(model, source_grid, final_target_grid, task_id, loss_mask, timestep, key):
    masked_target_grid = jnp.where(loss_mask, MASK_VALUE, final_target_grid)
    # The model expects a single example, which is what vmap provides.
    pred_logits = model(source_grid, masked_target_grid, task_id, timestep, key=key)
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, final_target_grid)
    masked_loss = (losses * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1)
    return masked_loss

@eqx.filter_jit
def train_step(model, optim_state, batch, key):
    def compute_batch_loss(model, batch, key):
        masking_keys, loss_keys, ratio_keys = jax.random.split(key, 3)
        masking_keys = jax.random.split(masking_keys, batch["source"].shape[0])
        loss_keys = jax.random.split(loss_keys, batch["source"].shape[0])
        
        random_ratios = jax.random.uniform(ratio_keys, shape=(batch["source"].shape[0],))
        
        def create_loss_mask(target, ratio, key):
            can_be_masked = (target >= 0) & (target < 10) 
            noise = jax.random.uniform(key, shape=target.shape)
            mask = (noise < ratio) & can_be_masked
            return mask

        loss_masks = jax.vmap(create_loss_mask)(batch["target"], random_ratios, masking_keys)

        vmapped_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0))
        per_example_losses = vmapped_loss_fn(
            model, batch["source"], batch["target"], batch["task_id"], loss_masks, random_ratios, loss_keys
        )
        return jnp.mean(per_example_losses)

    loss, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch, key)
    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optim_state, loss

def generate_with_diffusion(model, source_grid, task_id, key, num_steps, unmask_per_step):
    num_cells = source_grid.shape[0] * source_grid.shape[1]
    def diffusion_step(i, state):
        current_target, key = state
        timestep = jnp.array((num_steps - 1 - i) / num_steps)
        
        pred_logits = model(source_grid, current_target, task_id, timestep, key=None)
        
        probs = jax.nn.softmax(pred_logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1)
        confidence = -entropy
        masked_confidence = jnp.where(current_target == MASK_VALUE, confidence, -jnp.inf)
        _, unmask_indices_flat = jax.lax.top_k(masked_confidence.flatten(), k=unmask_per_step)
        unmask_mask_flat = jnp.zeros(num_cells, dtype=jnp.bool_).at[unmask_indices_flat].set(True)
        unmask_mask = unmask_mask_flat.reshape(source_grid.shape)
        predicted_colors = jnp.argmax(pred_logits, axis=-1)
        new_target = jnp.where(unmask_mask, predicted_colors, current_target)
        return new_target, key
    initial_target = jnp.full_like(source_grid, MASK_VALUE)
    initial_state = (initial_target, key)
    final_target, _ = jax.lax.fori_loop(0, num_steps, diffusion_step, initial_state)
    return final_target

def evaluate_loss(model, dataset, key, batch_size=32):
    model = eqx.tree_inference(model, value=True)
    total_loss = 0.
    total_count = dataset['source'].shape[0]
    if total_count == 0:
        return 0.0

    @eqx.filter_jit
    def eval_loss_step(batch, key):
        masking_keys, _, ratio_keys = jax.random.split(key, 3)
        masking_keys = jax.random.split(masking_keys, batch["source"].shape[0])
        random_ratios = jax.random.uniform(ratio_keys, shape=(batch["source"].shape[0],))
        def create_loss_mask(target, ratio, key):
            can_be_masked = (target >= 0) & (target < 10)
            noise = jax.random.uniform(key, shape=target.shape)
            mask = (noise < ratio) & can_be_masked
            return mask
        loss_masks = jax.vmap(create_loss_mask)(batch["target"], random_ratios, masking_keys)
        vmapped_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None))
        per_example_losses = vmapped_loss_fn(model, batch["source"], batch["target"], batch["task_id"], loss_masks, random_ratios, None)
        return jnp.mean(per_example_losses)

    for i in range(0, total_count, batch_size):
        start_index = i
        end_index = i + batch_size
        batch = jax.tree_util.tree_map(lambda x: x[start_index:end_index], dataset)
        key, step_key = jax.random.split(key)
        total_loss += eval_loss_step(batch, step_key) * batch["source"].shape[0]
    
    return total_loss / total_count

def evaluate_generation_accuracy(model, dataset, key, num_diffusion_steps, batch_size=32):
    model = eqx.tree_inference(model, value=True)
    perfect_count = 0
    total_correct_pixels = 0
    total_pixels = 0
    total_count = dataset['source'].shape[0]
    if total_count == 0:
        return 0.0, 0.0

    num_cells = MAX_GRID_DIM * MAX_GRID_DIM
    unmask_per_step = max(1, num_cells // num_diffusion_steps)
    vmapped_generate = eqx.filter_jit(jax.vmap(generate_with_diffusion, in_axes=(None, 0, 0, 0, None, None)))

    for i in range(0, total_count, batch_size):
        start_index = i
        end_index = i + batch_size
        batch = jax.tree_util.tree_map(lambda x: x[start_index:end_index], dataset)
        
        key, step_key = jax.random.split(key)
        batch_keys = jax.random.split(step_key, batch["source"].shape[0])
        
        generated_grids = vmapped_generate(model, batch["source"], batch["task_id"], batch_keys, num_diffusion_steps, unmask_per_step)
        matches = (generated_grids == batch["target"])
        
        perfect_count += jnp.all(matches, axis=(1, 2)).sum()
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
    NUM_EPOCHS = 5000
    NUM_DIFFUSION_STEPS = 100
    DROPOUT_RATE = 0.1
    BASE_CHANNELS = 256

    # --- Data Loading ---
    print("Loading and processing ARC data into grids...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")
    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}
    train_data_list = []
    eval_data_list = []
    for task_id, task in training_tasks[:NUM_TASKS]:
        task_num = task_id_to_number[task_id]
        for pair in task['train']:
            if pair["input"] is None or pair["output"] is None: continue
            train_data_list.append({ "source": pad_grid(np.array(pair['input'])), "target": pad_grid(np.array(pair['output'])), "task_id": task_num })
        for pair in task['test']:
            if pair["input"] is None or pair["output"] is None: continue
            eval_data_list.append({ "source": pad_grid(np.array(pair['input'])), "target": pad_grid(np.array(pair['output'])), "task_id": task_num })
    
    train_dataset_cpu = { "source": np.array([d['source'] for d in train_data_list]), "target": np.array([d['target'] for d in train_data_list]), "task_id": np.array([d['task_id'] for d in train_data_list])}
    eval_dataset_cpu = { "source": np.array([d['source'] for d in eval_data_list]), "target": np.array([d['target'] for d in eval_data_list]), "task_id": np.array([d['task_id'] for d in eval_data_list])}
    print(f"Loaded {train_dataset_cpu['source'].shape[0]} training pairs and {eval_dataset_cpu['source'].shape[0]} evaluation pairs.")

    # --- Model Initialization ---
    key = jax.random.PRNGKey(int(time.time()))
    model_key, train_key, eval_key, loader_key = jax.random.split(key, 4)
    model = ArcUNetSolver(num_tasks=NUM_TASKS, base_channels=BASE_CHANNELS, dropout_p=DROPOUT_RATE, key=model_key)
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def count_params(model: eqx.Module):
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model has {count_params(model):,} trainable parameters.")

    print(f"Training U-Net for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}...")
    num_train_samples = train_dataset_cpu['source'].shape[0]
    time_since_eval = time.time()
    
    for epoch in range(NUM_EPOCHS):
        loader_key, perm_key = jax.random.split(loader_key)
        indices = jax.random.permutation(perm_key, num_train_samples)
        
        shuffled_train_dataset_cpu = jax.tree_util.tree_map(lambda x: x[indices], train_dataset_cpu)
        
        for i in range(0, num_train_samples, BATCH_SIZE):
            start_index = i
            end_index = i + BATCH_SIZE
            batch_cpu = jax.tree_util.tree_map(lambda x: x[start_index:end_index], shuffled_train_dataset_cpu)
            if batch_cpu['source'].shape[0] != BATCH_SIZE:
                continue

            batch_gpu = jax.tree_util.tree_map(jnp.array, batch_cpu)

            train_key, step_key = jax.random.split(train_key)
            model, optim_state, train_loss = train_step(
                model, optim_state, batch_gpu, step_key
            )

        if time.time() - time_since_eval > 60 or epoch == NUM_EPOCHS - 1:
            time_since_eval = time.time()
            eval_key, train_loss_key, train_acc_key, eval_acc_key = jax.random.split(eval_key, 4)
            
            model_for_eval = eqx.tree_inference(model, value=True)
            
            train_dataset_gpu = jax.tree_util.tree_map(jnp.array, train_dataset_cpu)
            eval_dataset_gpu = jax.tree_util.tree_map(jnp.array, eval_dataset_cpu)

            train_loss_val = evaluate_loss(model_for_eval, train_dataset_gpu, train_loss_key, batch_size=BATCH_SIZE)
            eval_loss_val = evaluate_loss(model_for_eval, eval_dataset_gpu, eval_key, batch_size=BATCH_SIZE)
            
            train_acc, train_pixel_acc = evaluate_generation_accuracy(model_for_eval, train_dataset_gpu, train_acc_key, NUM_DIFFUSION_STEPS, batch_size=BATCH_SIZE)
            eval_acc, eval_pixel_acc = evaluate_generation_accuracy(model_for_eval, eval_dataset_gpu, eval_acc_key, NUM_DIFFUSION_STEPS, batch_size=BATCH_SIZE)

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | TL: {train_loss_val:.4f} | EL: {eval_loss_val:.4f} | TA: {train_acc:.2f}% ({train_pixel_acc:.2f}%) | EA: {eval_acc:.2f}% ({eval_pixel_acc:.2f}%)")

    # Final evaluation with the final model
    print("\n--- Final evaluation with final model ---")
    final_key = jax.random.PRNGKey(42)
    eval_dataset_gpu = jax.tree_util.tree_map(jnp.array, eval_dataset_cpu)
    final_acc, final_pixel_acc = evaluate_generation_accuracy(eqx.tree_inference(model, value=True), eval_dataset_gpu, final_key, NUM_DIFFUSION_STEPS, batch_size=BATCH_SIZE)
    print(f"Final Eval Perfect Solutions: {final_acc:.2f}% (Pixel Accuracy: {final_pixel_acc:.2f}%)")

