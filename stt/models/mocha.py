import torch
import torch.nn.functional as F


def moving_sum(x, back, forward):
    """Parallel moving sum with 1D Convolution"""
    # Pad window before applying convolution
    # [batch_size,    back + sequence_length + forward]
    x_padded = F.pad(x, pad=(back, forward))

    # Fake channel dimension for conv1d
    # [batch_size, 1, back + sequence_length + forward]
    x_padded = x_padded.unsqueeze(1)

    # Apply conv1d with filter of all ones for moving sum
    filters = x.new_ones(1, 1, back + forward + 1)

    x_sum = F.conv1d(x_padded, filters)

    # Remove fake channel dimension
    # [batch_size, sequence_length]
    return x_sum.squeeze(1)


def efficient_chunkwise_attention(chunk_size, emit_probs, softmax_logits):
    """Compute chunkwise attention distribution efficiently by clipping logits."""
    # Shift logits to avoid overflow
    softmax_logits -= torch.max(softmax_logits, 1, keepdims=True)
    # Limit the range for numerical stability
    softmax_exp = torch.exp(softmax_logits)
    softmax_exp = torch.clamp(softmax_exp, min=1e-5)
    # Compute chunkwise softmax denominators
    softmax_denominators = moving_sum(softmax_exp, chunk_size - 1, 0)
    # Compute \beta_{i, :}. emit_probs are \alpha_{i, :}.
    probs = softmax_exp * \
        moving_sum(emit_probs / softmax_denominators, 0, chunk_size - 1)
    return probs


def moving_max(x, w):
    """Compute the moving sum of x over a window with the provided bounds.

    x is expected to be of shape (batch_size, sequence_length).
    The returned tensor x_max is computed as
    x_max[i, j] = max(x[i, j - window + 1], ..., x[i, j])
    """
    # Pad x with -inf at the start
    # x = tf.pad(x, [[0, 0], [w - 1, 0]], mode='CONSTANT', constant_values=-np.inf)
    x = F.pad(x, pad=(w - 1, 0), value=-float('inf'))
    # Add "height" and "channel" dimensions (max_pool operates on 2D)
    # x = tf.reshape(x, [tf.shape(x)[0], 1, tf.shape(x)[1], 1])
    # x = tf.nn.max_pool(x, [1, 1, w, 1], [1, 1, 1, 1], 'VALID')
    x = x.unsqueeze(1)
    x = F.max_pool1d(x, kernel_size=w, stride=1).squeeze(1)
    # Remove "height" and "channel" dimensions
    return x


# def stable_chunkwise_attention(chunk_size, emit_probs, softmax_logits):
#     """Compute chunkwise attention distriobution stably by subtracting logit max."""
#     # Compute length-chunk_size sliding max of sequences in softmax_logits (m)
#     logits_max = moving_max(softmax_logits, chunk_size)

#     # Produce matrix with length-chunk_size frames of softmax_logits (D)
#     # Padding makes it so that the first frame is [-inf, -inf, ..., logits[0]]
#     # padded_logits = tf.pad(softmax_logits, [[0, 0], [chunk_size - 1, 0]],
#     #                      constant_values=-np.inf)
#     padded_logits = F.pad(
#         softmax_logits, pad=(chunk_size - 1, 0), value=-float('inf'))
#     framed_logits = tf.contrib.signal.frame(padded_logits, chunk_size, 1)
#     # Normalize each logit subsequence by the max in that subsequence
#     framed_logits = framed_logits - tf.expand_dims(logits_max, -1)
#     # Compute softmax denominators (d)
#     softmax_denominators = tf.reduce_sum(tf.exp(framed_logits), 2)
#     # Construct matrix of framed denominators, padding at the end so the final
#     # frame is [softmax_denominators[-1], inf, inf, ..., inf] (E)
#     framed_denominators = tf.contrib.signal.frame(
#         softmax_denominators, chunk_size, 1, pad_end=True, pad_value=np.inf)

#     # Create matrix of copied logits so that column j is softmax_logits[j] copied
#     # chunk_size times (N)
#     batch_size, seq_len = tf.unstack(tf.shape(softmax_logits))
#     copied_shape = (batch_size, seq_len, chunk_size)
#     copied_logits = (tf.expand_dims(softmax_logits, -1) *
#                      tf.ones(copied_shape, softmax_logits.dtype))
#     # Subtract the max over subsequences(M) from each logit
#     framed_max = tf.contrib.signal.frame(logits_max, chunk_size, 1,
#                                          pad_end=True, pad_value=np.inf)
#     copied_logits = copied_logits - framed_max
#     # Take exp() to get softmax numerators
#     softmax_numerators = tf.exp(copied_logits)

#     # Create matrix with length-chunk_size frames of emit_probs, padded so that
#     # the last frame is [emit_probs[-1], 0, 0, ..., 0] (A)
#     framed_probs = tf.contrib.signal.frame(
#         emit_probs, chunk_size, 1, pad_end=True)

#     # Compute chunkwise probability distributions
#     return tf.reduce_sum(framed_probs*softmax_numerators/framed_denominators, 2)
