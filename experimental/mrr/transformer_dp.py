import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, List, Tuple

from llm_python.utils.task_loader import TaskData, get_task_loader
import numpy as np

# --- 1. Tokenization and Data Processing ---

# Define special tokens for our sequence representation
NEWLINE_TOKEN = 10
PADDING_TOKEN = 11
VOCAB_SIZE = 12  # 0-9 for colors, 10 for newline, 11 for padding
MAX_SEQ_LEN = 30 * 30 + 30  # Max grid size + max newlines


def tokenize_grid(grid: np.ndarray) -> np.ndarray:
    """Flattens a 2D grid into a 1D sequence with newline delimiters."""
    h, w = grid.shape
    tokens = []
    for i in range(h):
        tokens.extend(grid[i, :w].tolist())
        if i < h - 1:
            tokens.append(NEWLINE_TOKEN)
    return np.array(tokens, dtype=np.int32)


def process_and_pad_token_sequence(tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pads a token sequence to MAX_SEQ_LEN and creates an attention mask."""
    padded_tokens = np.full((MAX_SEQ_LEN,), PADDING_TOKEN, dtype=np.int32)
    padded_tokens[: len(tokens)] = tokens

    # Attention mask is True for real tokens, False for padding
    mask = np.full((MAX_SEQ_LEN,), False, dtype=np.bool_)
    mask[: len(tokens)] = True

    return padded_tokens, mask


# --- 2. NEW ARCHITECTURE: Position-wise Feed-Forward using Conv1d ---


class PositionWiseFeedForward(eqx.Module):
    """A position-wise feed-forward network implemented with 1D convolutions."""

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, input_size: int, dropout_p: float, *, key: jax.Array):
        key1, key2 = jax.random.split(key)
        hidden_size = 4 * input_size
        self.conv1 = eqx.nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1, key=key1
        )
        self.conv2 = eqx.nn.Conv1d(
            in_channels=hidden_size, out_channels=input_size, kernel_size=1, key=key2
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x has shape (seq_len, embed_dim)
        # Conv1d expects (channels, length), so we transpose
        x = jnp.transpose(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        # Transpose back to (seq_len, embed_dim)
        return jnp.transpose(x)


# --- 3. The Transformer Encoder Layer (with new FFN) ---


class TransformerEncoderLayer(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    ffn: PositionWiseFeedForward  # Replaced MLP with our new module
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self, input_size: int, num_heads: int, dropout_p: float, *, key: jax.Array
    ):
        attn_key, ffn_key = jax.random.split(key)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=input_size,
            output_size=input_size,
            dropout_p=dropout_p,
            key=attn_key,
        )
        self.ffn = PositionWiseFeedForward(input_size, dropout_p=dropout_p, key=ffn_key)
        self.norm1 = eqx.nn.LayerNorm(shape=input_size)
        self.norm2 = eqx.nn.LayerNorm(shape=input_size)
        self.dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(
        self, x: jnp.ndarray, mask: jnp.ndarray, *, key: jax.Array
    ) -> jnp.ndarray:
        if mask is not None:
            seq_len = mask.shape[0]
            attention_bias_row = jnp.where(mask, 0.0, -jnp.inf)
            attention_bias = jnp.broadcast_to(attention_bias_row, (seq_len, seq_len))
        else:
            attention_bias = None

        attn_key, ffn_key = (None, None) if key is None else jax.random.split(key)

        attn_input = jax.vmap(self.norm1)(x)
        attn_output = self.attention(
            query=attn_input,
            key_=attn_input,
            value=attn_input,
            mask=attention_bias,
            key=attn_key,
        )
        x = x + attn_output

        ffn_input = jax.vmap(self.norm2)(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output, key=ffn_key)

        return x


# --- 4. The Main Transformer Solver Model ---


class ArcTransformerSolver(eqx.Module):
    task_embedder: eqx.nn.Embedding
    token_embedder: eqx.nn.Embedding
    pos_embedder: eqx.nn.Embedding
    transformer_layers: List[TransformerEncoderLayer]
    output_projection: eqx.nn.Linear
    final_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        num_tasks: int,
        task_embed_dim: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_layers + 4)

        self.task_embedder = eqx.nn.Embedding(num_tasks, embed_dim, key=keys[0])
        self.token_embedder = eqx.nn.Embedding(VOCAB_SIZE, embed_dim, key=keys[1])
        self.pos_embedder = eqx.nn.Embedding(MAX_SEQ_LEN, embed_dim, key=keys[2])

        self.transformer_layers = [
            TransformerEncoderLayer(
                input_size=embed_dim, num_heads=num_heads, dropout_p=0.1, key=k
            )
            for k in keys[3:-1]
        ]
        self.final_norm = eqx.nn.LayerNorm(shape=embed_dim)
        self.output_projection = eqx.nn.Linear(embed_dim, VOCAB_SIZE, key=keys[-1])

    def __call__(
        self,
        tokens: jnp.ndarray,
        task_id: jnp.ndarray,
        attention_mask: jnp.ndarray,
        *,
        key: jax.Array,
    ) -> jnp.ndarray:
        token_embeds = jax.vmap(self.token_embedder)(tokens)
        task_embed = self.task_embedder(task_id)
        positions = jnp.arange(MAX_SEQ_LEN)
        pos_embeds = jax.vmap(self.pos_embedder)(positions)

        x = token_embeds + pos_embeds + task_embed

        layer_keys = [None] * len(self.transformer_layers)
        if key is not None:
            layer_keys = jax.random.split(key, len(self.transformer_layers))

        for layer, k in zip(self.transformer_layers, layer_keys):
            x = layer(x, mask=attention_mask, key=k)

        x = jax.vmap(self.final_norm)(x)
        logits = jax.vmap(self.output_projection)(x)
        return logits


# --- 5. Training and Evaluation Logic ---


def loss_fn(
    model: ArcTransformerSolver,
    input_tokens: jnp.ndarray,
    target_tokens: jnp.ndarray,
    task_id: jnp.ndarray,
    mask: jnp.ndarray,
    key: jax.Array,
):
    pred_logits = model(input_tokens, task_id, mask, key=key)
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_tokens)
    masked_loss = (losses * mask).sum() / mask.sum()
    return masked_loss


@eqx.filter_jit
def train_step(
    model: ArcTransformerSolver,
    optim_state: optax.OptState,
    batch: dict,
    key: jax.Array,
):
    def compute_batch_loss(model, batch, key):
        keys = jax.random.split(key, batch["input"].shape[0])
        per_example_losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0))(
            model,
            batch["input"],
            batch["target"],
            batch["task_id"],
            batch["mask"],
            keys,
        )
        return jnp.mean(per_example_losses)

    loss, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch, key)
    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optim_state, loss


def eval_model(model, inputs, targets, task_ids, masks, key: jax.Array):
    total_loss = 0
    perfect_count = 0
    total_count = inputs.shape[0]
    model = eqx.tree_inference(model, value=True)
    BATCH_SIZE = 32

    # Define a simple helper function to handle the keyword argument for the vmapped model call
    def predict_fn(tokens, task_id, mask):
        # We are in inference mode, so the key is None
        return model(tokens, task_id, mask, key=None)

    for i in range(0, total_count, BATCH_SIZE):
        batch_inputs, batch_targets = (
            inputs[i : i + BATCH_SIZE],
            targets[i : i + BATCH_SIZE],
        )
        batch_task_ids, batch_masks = (
            task_ids[i : i + BATCH_SIZE],
            masks[i : i + BATCH_SIZE],
        )

        # Use the helper function with vmap
        pred_logits = jax.vmap(predict_fn, in_axes=(0, 0, 0))(
            batch_inputs, batch_task_ids, batch_masks
        )
        pred_tokens = jnp.argmax(pred_logits, axis=-1)

        batch_loss_vector = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, None))(
            model, batch_inputs, batch_targets, batch_task_ids, batch_masks, None
        )
        total_loss += batch_loss_vector.sum()

        matches = (pred_tokens == batch_targets) | (batch_masks == 0)
        perfect_count += jnp.all(matches, axis=1).sum()

    mean_loss = total_loss / total_count
    percent_perfect = 100.0 * perfect_count / total_count
    return mean_loss, percent_perfect


# --- 6. Main function to run the experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    NUM_TASKS = 400
    TASK_EMBED_DIM = 128
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 1000

    # Transformer Hyperparameters
    NUM_LAYERS = 4
    NUM_HEADS = 4
    EMBED_DIM = 128

    # --- Data Loading ---
    print("Loading and tokenizing ARC data...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")
    task_id_to_number = {task_id: i for i, (task_id, _) in enumerate(training_tasks)}

    def get_token_data(tasks, task_id_map, task_limit=None, split: str = "train"):
        inputs, targets, masks, task_nums = [], [], [], []
        for task_id, task in tasks if task_limit is None else tasks[:task_limit]:
            task_num = task_id_map[task_id]
            for pair in task[split]:
                if pair["input"] is None or pair["output"] is None:
                    continue

                input_tokens = tokenize_grid(np.array(pair["input"]))
                target_tokens = tokenize_grid(np.array(pair["output"]))

                padded_input, input_mask = process_and_pad_token_sequence(input_tokens)
                padded_target, _ = process_and_pad_token_sequence(target_tokens)

                inputs.append(padded_input)
                targets.append(padded_target)
                masks.append(input_mask)
                task_nums.append(task_num)

        return map(np.array, [inputs, targets, masks, task_nums])

    inputs_train, targets_train, masks_train, task_numbers_train = get_token_data(
        training_tasks, task_id_to_number, task_limit=NUM_TASKS, split="train"
    )
    inputs_eval, targets_eval, masks_eval, task_numbers_eval = get_token_data(
        training_tasks, task_id_to_number, task_limit=NUM_TASKS, split="test"
    )
    print(f"Total pairs: {len(inputs_train)}")

    # --- Model Initialization ---
    key = jax.random.PRNGKey(0)
    model = ArcTransformerSolver(
        num_tasks=NUM_TASKS,
        task_embed_dim=TASK_EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=EMBED_DIM,
        key=key,
    )
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def count_params(model: eqx.Module):
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        )

    print(f"Model has {count_params(model):,} trainable parameters.")

    # --- Training Loop ---
    print(f"Training for {NUM_EPOCHS} epochs...")
    train_key, eval_key, loader_key = jax.random.split(key, 3)
    num_train_samples = inputs_train.shape[0]

    time_since_eval = time.time()

    for epoch in range(NUM_EPOCHS):
        loader_key, _ = jax.random.split(loader_key)
        indices = jax.random.permutation(loader_key, num_train_samples)

        for i in range(0, num_train_samples, BATCH_SIZE):
            train_key, _ = jax.random.split(train_key)
            batch_indices = indices[i : i + BATCH_SIZE]
            batch = {
                "input": inputs_train[batch_indices],
                "target": targets_train[batch_indices],
                "task_id": task_numbers_train[batch_indices],
                "mask": masks_train[batch_indices],
            }
            model, optim_state, train_loss = train_step(
                model, optim_state, batch, train_key
            )

        if time.time() - time_since_eval > 30:
            time_since_eval = time.time()
            eval_key, _ = jax.random.split(eval_key)
            train_loss, train_acc = eval_model(
                model,
                inputs_train,
                targets_train,
                task_numbers_train,
                masks_train,
                eval_key,
            )
            eval_loss, eval_acc = eval_model(
                model,
                inputs_eval,
                targets_eval,
                task_numbers_eval,
                masks_eval,
                eval_key,
            )
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} Train Loss: {train_loss:.4g} Train Acc: {train_acc:.4g}% Eval Loss: {eval_loss:.4g} Eval Acc: {eval_acc:.4g}%"
            )
