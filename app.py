import os
import functools
import time

import chess
import chess.svg
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from flask import Flask, request, jsonify

# Searchless Chess imports (assuming they are importable from /app/searchless_chess)
try:
    from searchless_chess.src import tokenizer
    from searchless_chess.src import training_utils
    from searchless_chess.src import transformer
    from searchless_chess.src import utils
    from searchless_chess.src.engines import engine
    from searchless_chess.src.engines import neural_engines
except ImportError as e:
    print(f"Error importing searchless_chess modules: {e}")
    print("Ensure the searchless_chess directory is in the Python path or structured correctly.")
    # You might need to add '/app' to PYTHONPATH if running from /app
    import sys
    sys.path.insert(0, '/app/searchless_chess')
    from searchless_chess.src import tokenizer
    from searchless_chess.src import training_utils
    from searchless_chess.src import transformer
    from searchless_chess.src import utils
    from searchless_chess.src.engines import engine
    from searchless_chess.src.engines import neural_engines


app = Flask(__name__)

# --- Global Variables for Model and Engine ---
# These will be loaded once when the Flask app starts
predictor = None
params = None
neural_engine = None
get_embedding_fn = None
predictor_config = None # Store config for embedding extractor
policy = None # Store policy

# --- Helper functions from your notebook (for embedding) ---
# (Copied directly from your provided code, ensure imports like jnp are correct)
def layer_norm(x: jax.Array) -> jax.Array:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

def shift_right(sequences: jax.Array) -> jax.Array:
    bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
    padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
    return padded_sequences[:, :-1]

def _mlp_block(inputs: jax.Array, config: transformer.TransformerConfig) -> jax.Array:
    ffn_dim = config.embedding_dim * config.widening_factor
    split_1 = hk.Linear(ffn_dim, with_bias=False)(inputs)
    split_2 = hk.Linear(ffn_dim, with_bias=False)(inputs)
    gate_output = jnn.silu(split_1) * split_2
    return hk.Linear(config.embedding_dim, with_bias=False)(gate_output)

def _attention_block(inputs: jax.Array, config: transformer.TransformerConfig) -> jax.Array:
    batch_size, sequence_length = inputs.shape[:2]
    if config.use_causal_mask:
        # Using jnp.tril for JAX compatibility within jit
        causal_mask = jnp.tril(
            jnp.ones((batch_size, 1, sequence_length, sequence_length))
        )
    else:
        causal_mask = None

    block = transformer.MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
        apply_qk_layernorm=config.apply_qk_layernorm,
    )
    return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask)

def transformer_embedding_extractor(
    targets: jax.Array,
    config: transformer.TransformerConfig,
) -> jax.Array:
    inputs = shift_right(targets)
    # Assuming embed_sequences is accessible via transformer module
    embeddings = transformer.embed_sequences(inputs, config)
    h = embeddings
    for _ in range(config.num_layers):
        attention_input = layer_norm(h)
        attention = _attention_block(attention_input, config)
        h += attention
        mlp_input = layer_norm(h)
        mlp_output = _mlp_block(mlp_input, config)
        h += mlp_output
    if config.apply_post_ln:
        h = layer_norm(h)
    return h # Shape [B, T, embedding_dim]

# --- Initialization Function ---
def load_model():
    global predictor, params, neural_engine, get_embedding_fn, predictor_config, policy

    if predictor is not None: # Already loaded
        print("Model already loaded.")
        return

    print("Loading model...")
    start_time = time.time()

    # --- Configuration from Environment Variables or defaults ---
    model_name = os.getenv('MODEL_NAME', '9M') # Default to 9M
    policy = os.getenv('POLICY', 'action_value') # Default policy
    num_return_buckets = int(os.getenv('NUM_RETURN_BUCKETS', '128'))
    print(f"Using Model: {model_name}, Policy: {policy}")

    match policy:
        case 'action_value':
            output_size = num_return_buckets
        case 'behavioral_cloning':
            output_size = utils.NUM_ACTIONS
        case 'state_value':
            output_size = num_return_buckets
        case _:
            raise ValueError(f'Unknown policy: {policy}')

    # --- Model parameter mapping ---
    if model_name == '9M':
        num_layers = 8
        embedding_dim = 256
        num_heads = 8
        widening_factor = 4 # Common default, check if defined elsewhere
    elif model_name == '136M':
        num_layers = 8
        embedding_dim = 1024
        num_heads = 8
        widening_factor = 4
    elif model_name == '270M':
        num_layers = 16
        embedding_dim = 1024
        num_heads = 8
        widening_factor = 4
    else: # Add more or raise error
        raise ValueError(f"Unknown pretrained model name: {model_name}")

    predictor_config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        widening_factor=widening_factor, # Added widening_factor
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False, # Important for standard transformer encoder behavior
    )

    # --- Build Predictor ---
    predictor = transformer.build_transformer_predictor(config=predictor_config)

    # --- Load Parameters ---
    # Adjust path relative to Dockerfile WORKDIR /app
    checkpoint_dir = os.path.join(
        '/app/searchless_chess/checkpoints/', model_name
    )
    print(f"Attempting to load checkpoints from: {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}. Did download.sh run correctly?")

    # Create dummy params with the correct structure
    dummy_params = predictor.initial_params(
        rng=jax.random.PRNGKey(0),
        targets=np.zeros((1, 1), dtype=np.uint32),
    )

    # Load real parameters
    params = training_utils.load_parameters(
        checkpoint_dir=checkpoint_dir,
        params=dummy_params,
        use_ema_params=True,
        step=-1, # Load latest
    )

    # --- Create Engine ---
    predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1) # Batch size 1 for API
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(
        num_return_buckets
    )

    neural_engine = neural_engines.ENGINE_FROM_POLICY[policy](
        return_buckets_values=return_buckets_values,
        predict_fn=predict_fn,
        temperature=0.005, # Low temperature for deterministic best move
    )

    # --- Create Embedding Function ---
    embedding_model = hk.transform(functools.partial(transformer_embedding_extractor, config=predictor_config))

    # Jit the embedding function for performance
    @jax.jit
    def _get_embedding(current_params, sequences):
        # Provide a dummy RNG key if required by Haiku apply, although likely unused in inference
        # Using None might work depending on Haiku version/setup
        try:
             return embedding_model.apply(current_params, None, sequences)
        except TypeError: # If RNG is mandatory
             print("Using dummy RNG key for embedding model apply.")
             return embedding_model.apply(current_params, jax.random.PRNGKey(42), sequences)

    get_embedding_fn = _get_embedding # Assign the jitted function

    end_time = time.time()
    print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")


# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_move():
    """
    Predicts the best move(s) for a given FEN string.
    Input: JSON {'fen': 'fen_string'}
    Output: JSON {'best_move': 'uci', 'moves': [{'move': 'uci', 'score': float}, ...]} or {'error': 'message'}
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded yet'}), 503 # Service Unavailable

    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing "f en" in request body'}), 400

    fen_string = data['fen']
    try:
        board = chess.Board(fen_string)
    except ValueError:
        return jsonify({'error': 'Invalid FEN string'}), 400

    if board.is_game_over():
         return jsonify({'best_move': None, 'moves': [], 'analysis': 'Game over'})

    try:
        # Use analyse to get scores for all moves
        results = neural_engine.analyse(board)
        buckets_log_probs = results['log_probs'] # Shape (num_legal_moves, num_buckets)
        _, return_buckets_values = utils.get_uniform_buckets_edges_values(predictor_config.output_size)

        # Compute expected return (win probability approximation) for each move
        win_probs = np.inner(np.exp(buckets_log_probs), return_buckets_values)
        sorted_legal_moves = engine.get_ordered_legal_moves(board)

        # Create sorted list of moves and scores
        move_scores = []
        for i in np.argsort(win_probs)[::-1]: # Sort descending
            move = sorted_legal_moves[i]
            score = float(win_probs[i]) # Convert numpy float to python float
            move_scores.append({'move': move.uci(), 'score': score})

        best_move = move_scores[0]['move'] if move_scores else None

        return jsonify({'best_move': best_move, 'moves': move_scores})

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"Error during prediction for FEN {fen_string}: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/embedding', methods=['POST'])
def get_position_embedding():
    """
    Calculates the final hidden state embedding for a given FEN string.
    Input: JSON {'fen': 'fen_string'}
    Output: JSON {'embedding': [float, ...]} or {'error': 'message'}
    """
    if get_embedding_fn is None or params is None or predictor_config is None or policy is None:
        return jsonify({'error': 'Model or embedding function not loaded yet'}), 503

    data = request.get_json()
    if not data or 'fen' not in data:
        return jsonify({'error': 'Missing "fen" in request body'}), 400

    fen_string = data['fen']
    try:
        board = chess.Board(fen_string)
    except ValueError:
        return jsonify({'error': 'Invalid FEN string'}), 400

    try:
        # --- Prepare input sequence based on policy ---
        # (Logic adapted from your notebook cell)
        tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
        tokenized_fen = np.expand_dims(tokenized_fen, axis=0) # Shape (1, T_state)

        if policy == 'action_value':
            # Need a dummy action and return. Use a common one or zero? Let's use zeros.
            # Note: The 'correct' action might matter for the exact embedding used in THAT policy's prediction.
            # If you want a general state embedding, 'state_value' or 'behavioral_cloning' might be more appropriate conceptually.
            dummy_action = np.zeros((1, 1), dtype=np.int32)
            dummy_return = np.zeros((1, 1), dtype=np.int32)
            input_sequence = np.concatenate([tokenized_fen, dummy_action, dummy_return], axis=1)
            # The embedding extractor applies shift_right internally.
            # The 'final' embedding before the output layer corresponds to the last input token's position.
            # For action_value, this would typically correspond to the 'action' input token.

        elif policy == 'state_value':
            dummy_return = np.zeros((1, 1), dtype=np.int32)
            input_sequence = np.concatenate([tokenized_fen, dummy_return], axis=1)
            # Embedding at the final token of the state is used for value prediction.

        elif policy == 'behavioral_cloning':
            dummy_action = np.zeros((1, 1), dtype=np.int32)
            input_sequence = np.concatenate([tokenized_fen, dummy_action], axis=1)
            # Embedding at the final token of the state is used for action prediction.
        else:
             return jsonify({'error': f'Embedding extraction not implemented for policy: {policy}'}), 501


        # --- Get the full sequence embedding ---
        # embedding_output shape: (1, sequence_length_after_shift, embedding_dim)
        embedding_output = get_embedding_fn(params, input_sequence)

        # --- Extract the relevant embedding ---
        # Usually, the embedding corresponding to the *last token of the input sequence*
        # (before padding/prediction tokens) is considered the 'state' representation.
        # After shift_right, the input sequence has length `input_sequence.shape[1]`.
        # The output `h` from the transformer has shape [B, T, D] where T is this length.
        # We take the embedding at the last time step T-1.
        position_embedding = embedding_output[:, -1, :] # Shape: (1, embedding_dim)

        # Convert JAX array to list for JSON serialization
        position_embedding_list = np.array(position_embedding[0]).tolist()

        return jsonify({'embedding': position_embedding_list})

    except Exception as e:
        app.logger.error(f"Error during embedding calculation for FEN {fen_string}: {e}", exc_info=True)
        return jsonify({'error': f'Embedding calculation failed: {str(e)}'}), 500


# --- Load the model when the application starts ---
# Using @app.before_first_request is common, but it's deprecated in newer Flask versions.
# A simple approach is to call it directly, Flask handles concurrency.
# Or use app.app_context() for more complex setups.
with app.app_context():
    load_model()

# --- Main execution ---
if __name__ == '__main__':
    # Use Flask's development server
    # For production, use a WSGI server like Gunicorn (see Dockerfile CMD)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) # debug=False for production/docker