import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Sequence, Tuple, List

from llm_python.utils.task_loader import get_task_loader
import numpy as np

# --- 1. The Encoder Module (Unchanged) ---
# This module takes a grid and a task ID and compresses them into a single "thought vector".


class Encoder(nn.Module):
    """Encodes a 30x30 grid, conditioned by a task embedding, into a latent vector."""

    latent_dim: int = 256

    @nn.compact
    def __call__(self, grid: jnp.ndarray, task_embedding: jnp.ndarray) -> jnp.ndarray:
        # Input grid shape: (batch_size, 30, 30) with integer colors (0-9)
        # task_embedding shape: (batch_size, task_embed_dim)

        x = nn.one_hot(grid, num_classes=10)

        # The CNN layers are now modulated by the task embedding via FiLM layers.
        # This allows the feature extraction to be task-specific.
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = FiLMLayer()(x, task_embedding)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = FiLMLayer()(x, task_embedding)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = FiLMLayer()(x, task_embedding)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        # We no longer concatenate the raw task_id here. The visual features
        # are already task-conditioned thanks to the FiLM layers.
        latent_vector = nn.Dense(features=self.latent_dim, name="latent_projection")(x)
        return latent_vector


# --- 2. NEW: The FiLM Layer and Modified Refiner ---

class FiLMLayer(nn.Module):
    """A Feature-wise Linear Modulation Layer."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch, H, W, C)
        # conditioning shape: (batch, embed_dim)
        num_channels = x.shape[-1]
        
        # Project the conditioning vector to get per-channel scale (gamma) and shift (beta)
        projection = nn.Dense(features=num_channels * 2, name="film_projection")(conditioning)
        
        gamma = projection[..., :num_channels]
        beta = projection[..., num_channels:]
        
        # Reshape for broadcasting: (batch, C) -> (batch, 1, 1, C)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        
        # Apply the modulation
        return (gamma * x) + beta


class Refiner(nn.Module):
    """A simple CNN block that refines logits, conditioned by the task."""
    @nn.compact
    def __call__(self, logits: jnp.ndarray, task_embedding: jnp.ndarray) -> jnp.ndarray:
        residual = logits
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(logits)
        x = FiLMLayer()(x, task_embedding) # Condition the refinement process
        x = nn.relu(x)
        x = nn.Conv(features=10, kernel_size=(1, 1))(x) # Project back to 10 color channels
        return x + residual


# --- 3. MODIFIED: The Iterative Decoder ---

class IterativeDecoder(nn.Module):
    """Decodes a latent vector by iteratively refining its output grid."""
    num_refinement_steps: int = 8

    def setup(self):
        self.initial_projection = nn.Dense(features=3 * 3 * 128)
        self.upsampler = [
            nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="VALID"),
            nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="VALID"),
            nn.ConvTranspose(features=10, kernel_size=(4, 4), strides=(2, 2), padding="SAME"),
        ]
        self.refiner = Refiner()

    def __call__(self, latent_vector: jnp.ndarray, task_embedding: jnp.ndarray) -> List[jnp.ndarray]:
        # Step 1: Create the initial "rough sketch" grid.
        x = self.initial_projection(latent_vector)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 3, 3, 128))

        # Upsample it
        x = nn.relu(self.upsampler[0](x))
        x = nn.relu(self.upsampler[1](x))
        current_logits = self.upsampler[2](x)

        # Step 2: The Outer Loop for refinement.
        all_logits = [current_logits]
        for _ in range(self.num_refinement_steps):
            # Pass the task_embedding to the refiner at each step
            current_logits = self.refiner(current_logits, task_embedding)
            all_logits.append(current_logits)

        return all_logits


# --- 4. MODIFIED: The Full Autoencoder Model & Training Logic ---


class Autoencoder(nn.Module):
    """The full autoencoder model combining a task-conditioned Encoder and Decoder."""

    latent_dim: int
    num_tasks: int
    num_refinement_steps: int
    task_embed_dim: int = 64

    def setup(self):
        self.task_embedder = nn.Dense(features=self.task_embed_dim)
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = IterativeDecoder(num_refinement_steps=self.num_refinement_steps)

    def __call__(self, grid, task_id):
        # Create the task embedding first. This will be passed to both sub-modules.
        task_id_one_hot = nn.one_hot(task_id, num_classes=self.num_tasks)
        task_embedding = self.task_embedder(task_id_one_hot)

        # The encoder and decoder now both receive the conditioning vector.
        latent_vector = self.encoder(grid, task_embedding)
        reconstructed_logits_list = self.decoder(latent_vector, task_embedding)
        return reconstructed_logits_list


@jax.jit
def train_step(state: train_state.TrainState, batch: dict):
    """Performs a single training step with loss applied at each refinement step."""

    def loss_fn(params):
        # The model now returns a list of logit grids
        list_of_logits = state.apply_fn(
            {"params": params}, grid=batch["grid"], task_id=batch["task_id"]
        )

        total_loss = 0.0
        # Apply the masked loss to each prediction in the sequence
        for i, logits in enumerate(list_of_logits):
            per_pixel_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["grid"]
            )
            masked_loss = per_pixel_loss * batch["mask"]
            mean_loss_at_step = masked_loss.sum() / batch["mask"].sum()
            
            # We can optionally weight the losses, giving more importance to later steps
            # For simplicity, we'll weight them all equally for now.
            total_loss += mean_loss_at_step

        # Return the average loss across all refinement steps
        return total_loss / len(list_of_logits)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# --- 5. MODIFIED: Main function to run the experiment ---
def main():
    def get_batches(arrays, batch_size):
        n = arrays[0].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield tuple(arr[start:end] for arr in arrays)

    # --- Configuration ---
    LATENT_DIM = 256
    NUM_TASKS = 400
    TASK_EMBED_DIM = 64 # Add parameter for the new embedding
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_REFINEMENT_STEPS = 8
    NUM_EPOCHS = 1000

    key = jax.random.PRNGKey(0)
    init_key, _ = jax.random.split(key)

    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")

    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}
    
    grids, masks, task_numbers = [], [], []
    for task_id, task in training_tasks:
        for train in task["train"]:
            for grid in [train["input"], train["output"]]:
                task_numbers.append(task_id_to_number[task_id])
                grid = np.array(grid, dtype=np.int32)
                padded_grid = np.full((30, 30), 0, dtype=np.int32)
                h, w = grid.shape
                padded_grid[:h, :w] = grid
                grids.append(padded_grid)
                mask = np.zeros((30, 30), dtype=np.float32)
                mask[:h, :w] = 1.0
                masks.append(mask)
    grids = np.stack(grids)
    masks = np.stack(masks)
    task_numbers = np.array(task_numbers, dtype=np.int32)
    print(f"Loaded {len(grids)} total grids.")

    num_total = grids.shape[0]
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    split = int(num_total * 0.9)
    train_idx, eval_idx = indices[:split], indices[split:]
    grids_train, grids_eval = grids[train_idx], grids[eval_idx]
    masks_train, masks_eval = masks[train_idx], masks[eval_idx]
    task_numbers_train, task_numbers_eval = task_numbers[train_idx], task_numbers[eval_idx]
    print(f"Train grids: {grids_train.shape[0]}, Eval grids: {grids_eval.shape[0]}")

    model = Autoencoder(
        latent_dim=LATENT_DIM,
        num_tasks=NUM_TASKS,
        num_refinement_steps=NUM_REFINEMENT_STEPS,
        task_embed_dim=TASK_EMBED_DIM
    )
    params = model.init(init_key, grids_train, task_numbers_train)["params"]
    optimizer = optax.adam(LEARNING_RATE)
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    print(f"Training for {NUM_EPOCHS} epochs (batched, batch_size={BATCH_SIZE})...")
    num_train_samples = grids_train.shape[0]
    for epoch in range(NUM_EPOCHS):
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)
        grids_shuffled, task_numbers_shuffled, masks_shuffled = grids_train[indices], task_numbers_train[indices], masks_train[indices]
        batch_losses = []
        for batch_grids, batch_task_numbers, batch_masks in get_batches([grids_shuffled, task_numbers_shuffled, masks_shuffled], BATCH_SIZE):
            batch = {"grid": batch_grids, "task_id": batch_task_numbers, "mask": batch_masks}
            model_state, loss = train_step(model_state, batch)
            batch_losses.append(float(loss))
        mean_loss = np.mean(batch_losses)

        def get_eval_loss():
            list_of_logits = model.apply({"params": model_state.params}, grids_eval, task_numbers_eval)
            final_logits = list_of_logits[-1]
            per_pixel_loss = optax.softmax_cross_entropy_with_integer_labels(logits=final_logits, labels=grids_eval)
            masked_loss = per_pixel_loss * masks_eval
            mean_loss = masked_loss.sum() / masks_eval.sum()
            return float(mean_loss)
        eval_loss = get_eval_loss()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Mean Loss: {mean_loss:.4f} - Eval Mean Loss: {eval_loss:.4f}")

    def eval_reconstruction_stats(grids_eval, masks_eval, task_numbers_eval, split_name):
        list_of_logits = model.apply({"params": model_state.params}, grids_eval, task_numbers_eval)
        # For evaluation, we only care about the final, most refined prediction
        final_logits = list_of_logits[-1]
        
        per_pixel_loss = optax.softmax_cross_entropy_with_integer_labels(logits=final_logits, labels=grids_eval)
        masked_loss = per_pixel_loss * masks_eval
        mean_loss = masked_loss.sum() / masks_eval.sum()
        
        pred_grids = np.array(jnp.argmax(final_logits, axis=-1))
        perfect_count = 0
        total = pred_grids.shape[0]
        for i in range(total):
            match = (pred_grids[i] == grids_eval[i]) | (masks_eval[i] == 0)
            if np.all(match):
                perfect_count += 1
        percent_perfect = 100.0 * perfect_count / total
        print(f"\n--- {split_name} Reconstruction Stats ---")
        print(f"Mean loss: {float(mean_loss):.4f}")
        print(f"Perfectly reconstructed grids: {perfect_count}/{total} ({percent_perfect:.2f}%)")
        print(f"Shape of final reconstructed logits: {final_logits.shape}")

    eval_reconstruction_stats(grids_train, masks_train, task_numbers_train, "Train Set")
    eval_reconstruction_stats(grids_eval, masks_eval, task_numbers_eval, "Eval Set")

if __name__ == "__main__":
    main()


