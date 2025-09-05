import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List, Tuple

# Assuming 'llm_python' is a local module you have.
# Replace this with your actual ARC data loading utility.
from llm_python.utils.task_loader import get_task_loader
import numpy as np

# --- 1. Constants and Configuration ---

# Define palette and special values for the grid
MASK_VALUE = 10
VOCAB_SIZE = 11  # 0-9 for colors, 10 for mask
MAX_GRID_DIM = 30 # Pad all grids to 30x30

# --- 2. U-Net Architecture ---

class ConvBlock(eqx.Module):
    """A standard block of Conv -> GroupNorm -> Activation."""
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    
    def __init__(self, in_channels, out_channels, kernel_size=3, *, key: jax.Array):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, key=key)
        self.norm = eqx.nn.GroupNorm(groups=8, channels=out_channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # This block now expects a single (C, H, W) input
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.silu(x)
        return x

class ArcUNetSolver(eqx.Module):
    """A U-Net model conditioned on task ID for solving ARC puzzles."""
    task_embedder: eqx.nn.Embedding
    
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
        task_embed_dim: int,
        base_channels: int = 64,
        *,
        key: jax.Array
    ):
        keys = jax.random.split(key, 10)
        
        self.task_embedder = eqx.nn.Embedding(num_tasks, task_embed_dim, key=keys[0])
        
        in_ch = 2 * VOCAB_SIZE + task_embed_dim
        c1 = base_channels
        c2 = base_channels * 2

        self.down_block1 = [ConvBlock(in_ch, c1, key=keys[1]), ConvBlock(c1, c1, key=keys[2])]
        self.down_block2 = [ConvBlock(c1, c2, key=keys[3]), ConvBlock(c2, c2, key=keys[4])]
        self.mid_block1 = ConvBlock(c2, c2, key=keys[5])
        self.mid_block2 = ConvBlock(c2, c2, key=keys[6])
        self.up_block1 = [ConvBlock(c2 + c2, c1, key=keys[7]), ConvBlock(c1, c1, key=keys[8])]
        self.up_block2 = [ConvBlock(c1 + c1, c1, key=keys[9])]
        self.final_conv = eqx.nn.Conv2d(c1, VOCAB_SIZE, kernel_size=1, key=keys[9])
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, source_grid: jnp.ndarray, masked_target_grid: jnp.ndarray, task_id: jnp.ndarray) -> jnp.ndarray:
        # This model now expects single examples: (H, W) grids and a scalar task_id
        
        source_onehot = jax.nn.one_hot(source_grid, VOCAB_SIZE)
        masked_target_onehot = jax.nn.one_hot(masked_target_grid, VOCAB_SIZE)
        
        source_onehot = jnp.transpose(source_onehot, (2, 0, 1))
        masked_target_onehot = jnp.transpose(masked_target_onehot, (2, 0, 1))

        task_embed = self.task_embedder(task_id)
        task_cond = jnp.broadcast_to(task_embed[:, None, None], (task_embed.shape[0], MAX_GRID_DIM, MAX_GRID_DIM))
        
        x = jnp.concatenate([source_onehot, masked_target_onehot, task_cond], axis=0)

        skip1 = self.down_block1[1](self.down_block1[0](x))
        x = self.pool(skip1)

        skip2 = self.down_block2[1](self.down_block2[0](x))
        x = self.pool(skip2)

        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        x = jax.image.resize(x, (x.shape[0],) + skip2.shape[1:], "nearest")
        x = jnp.concatenate([x, skip2], axis=0)
        x = self.up_block1[1](self.up_block1[0](x))

        x = jax.image.resize(x, (x.shape[0],) + skip1.shape[1:], "nearest")
        x = jnp.concatenate([x, skip1], axis=0)
        x = self.up_block2[0](x)
        
        logits = self.final_conv(x)
        return jnp.transpose(logits, (1, 2, 0))

# --- 3. Data Pipeline for Grid-Based Diffusion ---

def pad_grid(grid: np.ndarray, max_dim: int = MAX_GRID_DIM) -> np.ndarray:
    h, w = grid.shape
    padded_grid = np.full((max_dim, max_dim), 0, dtype=np.int32)
    padded_grid[:h, :w] = grid
    return padded_grid

# --- 4. Training and Evaluation Logic ---

def loss_fn(model, source_grid, final_target_grid, task_id, loss_mask):
    masked_target_grid = jnp.where(loss_mask, MASK_VALUE, final_target_grid)
    pred_logits = model(source_grid, masked_target_grid, task_id)
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, final_target_grid)
    masked_loss = (losses * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1)
    return masked_loss

@eqx.filter_jit
def train_step(model, optim_state, batch, key, mask_ratio):
    
    def compute_batch_loss(model, batch, key):
        masking_keys = jax.random.split(key, batch["source"].shape[0])
        
        def create_loss_mask(target, key):
            can_be_masked = (target >= 0) & (target < 10) 
            k = int(target.size * mask_ratio)
            rand = jax.random.uniform(key, shape=target.shape)
            masked_positions = jnp.where(can_be_masked, rand, 1e9)
            _, flat_indices = jax.lax.top_k(-masked_positions.flatten(), k=k)
            loss_mask_flat = jnp.zeros_like(target.flatten(), dtype=jnp.bool_).at[flat_indices].set(True)
            return loss_mask_flat.reshape(target.shape)

        loss_masks = jax.vmap(create_loss_mask)(batch["target"], masking_keys)
        
        per_example_losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0))(
            model, batch["source"], batch["target"], batch["task_id"], loss_masks
        )
        return jnp.mean(per_example_losses)

    loss, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch, key)
    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optim_state, loss


def eval_model(model, task_data, task_id_map, key, mask_ratio=0.5, batch_size=32):
    model = eqx.tree_inference(model, value=True)
    
    eval_data = []
    for task_id, task in task_data:
        task_num = task_id_map[task_id]
        for pair in task['test']:
            if pair["input"] is None or pair["output"] is None: continue
            eval_data.append({
                "source": pad_grid(np.array(pair['input'])),
                "target": pad_grid(np.array(pair['output'])),
                "task_id": task_num
            })

    total_loss = 0.
    perfect_count = 0
    total_count = len(eval_data)
    if total_count == 0:
        return 0.0, 0.0

    # Define a function for a single example's evaluation logic
    def eval_single(source, target, task_id, key):
        masking_key, model_key = jax.random.split(key)
        
        can_be_masked = (target >= 0) & (target < 10) 
        k = int(target.size * mask_ratio)
        rand = jax.random.uniform(masking_key, shape=target.shape)
        masked_positions = jnp.where(can_be_masked, rand, 1e9)
        _, flat_indices = jax.lax.top_k(-masked_positions.flatten(), k=k)
        loss_mask_flat = jnp.zeros_like(target.flatten(), dtype=jnp.bool_).at[flat_indices].set(True)
        loss_mask = loss_mask_flat.reshape(target.shape)
        
        masked_target = jnp.where(loss_mask, MASK_VALUE, target)
        pred_logits = model(source, masked_target, task_id)
        pred_grid = jnp.argmax(pred_logits, axis=-1)
        
        losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, target)
        loss = (losses * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1)
        
        matches = (pred_grid == target) | (loss_mask == False)
        is_perfect = jnp.all(matches)
        
        return loss, is_perfect

    # JIT the vmapped version of the evaluation function
    vmapped_eval = eqx.filter_jit(jax.vmap(eval_single, in_axes=(0, 0, 0, 0)))

    for i in range(0, total_count, batch_size):
        raw_batch = eval_data[i : i + batch_size]
        batch = {
            "source": jnp.array([d['source'] for d in raw_batch]),
            "target": jnp.array([d['target'] for d in raw_batch]),
            "task_id": jnp.array([d['task_id'] for d in raw_batch]),
        }
        
        key, step_key = jax.random.split(key)
        batch_keys = jax.random.split(step_key, batch["source"].shape[0])

        batch_losses, batch_perfects = vmapped_eval(
            batch["source"], batch["target"], batch["task_id"], batch_keys
        )
        
        total_loss += batch_losses.sum()
        perfect_count += batch_perfects.sum()

    mean_loss = total_loss / total_count
    percent_perfect = 100.0 * perfect_count / total_count
    return mean_loss, percent_perfect

# --- 5. Main function to run the experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_TASKS = 400
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 2000
    MASK_RATIO = 0.75

    # --- Data Loading ---
    print("Loading and processing ARC data into grids...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")
    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}
    
    train_data = []
    for task_id, task in training_tasks[:NUM_TASKS]:
        task_num = task_id_to_number[task_id]
        for pair in task['train']:
            if pair["input"] is None or pair["output"] is None: continue
            train_data.append({
                "source": pad_grid(np.array(pair['input'])),
                "target": pad_grid(np.array(pair['output'])),
                "task_id": task_num
            })
    print(f"Loaded {len(train_data)} training pairs.")

    # --- Model Initialization ---
    key = jax.random.PRNGKey(int(time.time()))
    model_key, train_key, eval_key, loader_key = jax.random.split(key, 4)
    model = ArcUNetSolver(num_tasks=NUM_TASKS, task_embed_dim=64, base_channels=64, key=model_key)
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Training Loop ---
    print(f"Training U-Net for {NUM_EPOCHS} epochs with mask ratio {MASK_RATIO}...")
    num_train_samples = len(train_data)
    time_since_eval = time.time()

    for epoch in range(NUM_EPOCHS):
        loader_key, perm_key = jax.random.split(loader_key)
        indices = jax.random.permutation(perm_key, num_train_samples)
        
        for i in range(0, num_train_samples, BATCH_SIZE):
            batch_indices = indices[i : i + BATCH_SIZE]
            raw_batch = [train_data[j] for j in batch_indices]
            
            batch = {
                "source": jnp.array([d['source'] for d in raw_batch]),
                "target": jnp.array([d['target'] for d in raw_batch]),
                "task_id": jnp.array([d['task_id'] for d in raw_batch]),
            }
            
            train_key, step_key = jax.random.split(train_key)
            model, optim_state, train_loss = train_step(model, optim_state, batch, step_key, MASK_RATIO)

        if time.time() - time_since_eval > 20 or epoch == NUM_EPOCHS - 1:
            time_since_eval = time.time()
            eval_key, subkey = jax.random.split(eval_key)
            eval_loss, eval_acc = eval_model(model, training_tasks[:NUM_TASKS], task_id_to_number, subkey, mask_ratio=1.0, batch_size=BATCH_SIZE)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Step Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Eval Perfect Recon: {eval_acc:.2f}%")

