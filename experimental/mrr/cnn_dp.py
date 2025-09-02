import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Tuple

from llm_python.utils.task_loader import TaskData, get_task_loader
import numpy as np

# --- 1. The FiLM Layer (Equinox Version) ---
# This layer is key to making our convolutions task-specific.


class FiLMLayer(eqx.Module):
    """A Feature-wise Linear Modulation Layer in Equinox."""

    projection: eqx.nn.Linear

    def __init__(self, in_features: int, out_features: int, *, key: jax.random.PRNGKey):
        # This linear layer learns to produce the scale (gamma) and shift (beta)
        # parameters from the task conditioning vector.
        self.projection = eqx.nn.Linear(in_features, out_features * 2, key=key)

    def __call__(self, x: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
        # x shape: (H, W, C)
        # conditioning shape: (embed_dim,)

        # Get gamma and beta from the conditioning vector
        proj = self.projection(conditioning)
        num_channels = x.shape[-1]
        gamma = proj[:num_channels]
        beta = proj[num_channels:]

        # Apply the feature-wise modulation
        return (gamma * x) + beta


# --- 2. The Main Solver Model (Upgraded) ---


class ArcSolver(eqx.Module):
    """A task-conditioned, purely convolutional CNN to solve ARC tasks."""

    task_embedder: eqx.nn.Embedding
    conv_layers: List[eqx.nn.Conv2d]
    film_layers: List[FiLMLayer]

    def __init__(
        self, num_tasks: int, task_embed_dim: int, num_features: int, *, key: jax.Array
    ):
        # Keys for initializing layers
        keys = jax.random.split(key, 6)
        (
            embed_key,
            conv1_key,
            film1_key,
            conv2_key,
            film2_key,
            final_conv_key,
        ) = keys

        self.task_embedder = eqx.nn.Embedding(num_tasks, task_embed_dim, key=embed_key)

        in_channels = 10 * num_features
        # Significantly increased channel sizes for more capacity
        conv1_channels = 128
        conv2_channels = 256

        self.conv_layers = [
            eqx.nn.Conv2d(
                in_channels, conv1_channels, kernel_size=7, padding="SAME", key=conv1_key
            ),
            eqx.nn.Conv2d(
                conv1_channels, conv2_channels, kernel_size=7, padding="SAME", key=conv2_key
            ),
            eqx.nn.Conv2d(
                conv2_channels, 10, kernel_size=7, padding="SAME", key=final_conv_key
            ),
        ]
        self.film_layers = [
            FiLMLayer(task_embed_dim, conv1_channels, key=film1_key),
            FiLMLayer(task_embed_dim, conv2_channels, key=film2_key),
        ]

    def __call__(self, grid: jnp.ndarray, task_id: jnp.ndarray) -> jnp.ndarray:
        # grid shape: (H, W), task_id shape: () integer

        # --- Feature Generation ---
        features = jax.nn.one_hot(grid, num_classes=10)  # -> (H, W, 10)
        x = features

        # --- Purely Convolutional Path ---
        # Transpose to channels-first format for Equinox convolutions: (H, W, C) -> (C, H, W)
        x = jnp.transpose(x, (2, 0, 1))
        task_embedding = self.task_embedder(task_id)

        # Block 1
        x = self.conv_layers[0](x)
        x = jax.nn.relu(x)
        x = jnp.transpose(x, (1, 2, 0)) # -> (H, W, C)
        x = self.film_layers[0](x, task_embedding)
        x = jnp.transpose(x, (2, 0, 1)) # -> (C, H, W)

        # Block 2
        x = self.conv_layers[1](x)
        x = jax.nn.relu(x)
        x = jnp.transpose(x, (1, 2, 0)) # -> (H, W, C)
        x = self.film_layers[1](x, task_embedding)
        x = jnp.transpose(x, (2, 0, 1)) # -> (C, H, W)

        # --- Final Prediction ---
        logits = self.conv_layers[2](x)
        return jnp.transpose(logits, (1, 2, 0))


# --- 3. Training and Evaluation Logic ---


def process_grid(grid_data: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pads a grid to 30x30 and creates its corresponding mask."""
    grid = np.array(grid_data, dtype=np.int32)
    padded_grid = np.full((30, 30), 0, dtype=np.int32)
    h, w = grid.shape
    padded_grid[:h, :w] = grid
    mask = np.zeros((30, 30), dtype=np.float32)
    mask[:h, :w] = 1.0
    return padded_grid, mask


def get_grid_data(
    tasks: List[Tuple[str, TaskData]], task_id_map: Dict[str, int], task_limit:int = None
) -> Tuple[np.ndarray, ...]:
    """Processes raw task data into numpy arrays for training and evaluation."""
    inputs_train_list, outputs_train_list, masks_train_list, task_numbers_train_list = (
        [],
        [],
        [],
        [],
    )
    inputs_eval_list, outputs_eval_list, masks_eval_list, task_numbers_eval_list = (
        [],
        [],
        [],
        [],
    )

    for task_id, task in (tasks if task_limit is None else tasks[:task_limit]):
        task_num = task_id_map[task_id]
        for pair in task["train"]:
            if pair["input"] is None or pair["output"] is None:
                continue
            input_grid, _ = process_grid(pair["input"])
            output_grid, output_mask = process_grid(pair["output"])
            inputs_train_list.append(input_grid)
            outputs_train_list.append(output_grid)
            masks_train_list.append(output_mask)
            task_numbers_train_list.append(task_num)

        for pair in task["test"]:
            if pair["input"] is None or pair["output"] is None:
                continue
            input_grid, _ = process_grid(pair["input"])
            output_grid, output_mask = process_grid(pair["output"])
            inputs_eval_list.append(input_grid)
            outputs_eval_list.append(output_grid)
            masks_eval_list.append(output_mask)
            task_numbers_eval_list.append(task_num)

    # Convert lists to numpy arrays at the end
    return (
        np.array(inputs_train_list, dtype=np.int32),
        np.array(outputs_train_list, dtype=np.int32),
        np.array(masks_train_list, dtype=np.float32),
        np.array(task_numbers_train_list, dtype=np.int32),
        np.array(inputs_eval_list, dtype=np.int32),
        np.array(outputs_eval_list, dtype=np.int32),
        np.array(masks_eval_list, dtype=np.float32),
        np.array(task_numbers_eval_list, dtype=np.int32),
    )


def loss_fn(
    model: ArcSolver,
    input_grid: jnp.ndarray,
    target_grid: jnp.ndarray,
    task_id: jnp.ndarray,
    mask: jnp.ndarray,
):
    """Calculates the masked cross-entropy loss for a single input/output pair."""
    # The model is called with the INPUT grid
    pred_logits = model(input_grid, task_id)
    # The loss is calculated against the TARGET grid
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_grid)
    # The mask corresponds to the valid area of the TARGET grid
    masked_loss = (losses * mask).sum() / mask.sum()
    return masked_loss


@eqx.filter_jit
def train_step(model: ArcSolver, optim_state: optax.OptState, batch: dict):
    """Performs a single, batched training step."""

    def compute_batch_loss(model, batch):
        per_example_losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0))(
            model, batch["input"], batch["target"], batch["task_id"], batch["mask"]
        )
        return jnp.mean(per_example_losses)

    loss, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch)

    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_optim_state, loss


def eval_model(model, inputs, targets, task_ids, masks):
    """Evaluates the model's performance on a dataset."""
    total_loss = 0
    perfect_count = 0
    total_count = inputs.shape[0]
    
    # Define BATCH_SIZE for evaluation to avoid OOM errors
    BATCH_SIZE = 64 

    for i in range(0, total_count, BATCH_SIZE):
        batch_inputs = inputs[i : i + BATCH_SIZE]
        batch_targets = targets[i : i + BATCH_SIZE]
        batch_task_ids = task_ids[i : i + BATCH_SIZE]
        batch_masks = masks[i : i + BATCH_SIZE]

        pred_logits = jax.vmap(model)(batch_inputs, batch_task_ids)
        pred_grids = jnp.argmax(pred_logits, axis=-1)

        batch_loss_vector = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0))(
            model, batch_inputs, batch_targets, batch_task_ids, batch_masks
        )
        total_loss += batch_loss_vector.sum()

        matches = (pred_grids == batch_targets) | (batch_masks == 0)
        perfect_count += jnp.all(matches, axis=(1, 2)).sum()

    mean_loss = total_loss / total_count
    percent_perfect = 100.0 * perfect_count / total_count
    return mean_loss, percent_perfect


# --- 4. Main function to run the experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_TASKS = 20
    TASK_EMBED_DIM = 64
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5000

    # --- Data Loading ---
    print("Loading and processing ARC data...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")

    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}

    (
        inputs_train,
        outputs_train,
        masks_train,
        task_numbers_train,
        inputs_eval,
        outputs_eval,
        masks_eval,
        task_numbers_eval,
    ) = get_grid_data(training_tasks, task_id_to_number, task_limit=NUM_TASKS)

    print(f"Train pairs: {len(inputs_train)}, Eval pairs: {len(inputs_eval)}")

    # --- Model Initialization ---
    key = jax.random.PRNGKey(0)
    model = ArcSolver(
        num_tasks=NUM_TASKS, task_embed_dim=TASK_EMBED_DIM, num_features=1, key=key
    )
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))


    # --- Training Loop ---
    print(f"Training for {NUM_EPOCHS} epochs...")
    num_train_samples = inputs_train.shape[0]
    for epoch in range(NUM_EPOCHS):
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)

        for i in range(0, num_train_samples, BATCH_SIZE):
            batch_indices = indices[i : i + BATCH_SIZE]
            batch = {
                "input": inputs_train[batch_indices],
                "target": outputs_train[batch_indices],
                "task_id": task_numbers_train[batch_indices],
                "mask": masks_train[batch_indices],
            }
            model, optim_state, _ = train_step(model, optim_state, batch)

        # --- Evaluation at end of epoch ---
        if (epoch + 1) % 100 == 0:
            train_loss, train_acc = eval_model(
                model, inputs_train, outputs_train, task_numbers_train, masks_train
            )
            eval_loss, eval_acc = eval_model(
                model, inputs_eval, outputs_eval, task_numbers_eval, masks_eval
            )
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Eval Loss: {eval_loss:.4f} Eval Acc: {eval_acc:.2f}%"
            )

