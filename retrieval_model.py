import pickle
import numbers
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging


def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005), scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def embedding_loss(im_embeds, sent_embeds, im_labels, args):
    """
        im_embeds: (b, 512) image embedding tensors
        sent_embeds: (sample_size * b, 512) sentence embedding tensors
            where the order of sentence corresponds to the order of images and
            setnteces for the same image are next to each other
        im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
            True if and only if sentence[i], image[j] is a positive pair
    """
    # compute embedding loss
    sent_im_ratio = args.sample_size
    num_img = args.batch_size
    num_sent = num_img * sent_im_ratio

    sent_im_dist = pdist(sent_embeds, im_embeds)
    # image loss: sentence, positive image, and negative image
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, im_labels), [num_sent, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~im_labels), [num_sent, -1])
    im_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=args.num_neg_sample)[0])
    # sentence loss: image, positive sentence, and negative sentence
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.transpose(im_labels)), [num_img, -1])
    neg_pair_dist = tf.reshape(tf.tile(neg_pair_dist, [1, sent_im_ratio]), [num_sent, -1])
    sent_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=args.num_neg_sample)[0])
    # sentence only loss (neighborhood-preserving constraints)
    sent_sent_dist = pdist(sent_embeds, sent_embeds)
    sent_sent_mask = tf.reshape(tf.tile(tf.transpose(im_labels), [1, sent_im_ratio]), [num_sent, num_sent])
    pos_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, sent_sent_mask), [-1, sent_im_ratio])
    pos_pair_dist = tf.reduce_max(pos_pair_dist, axis=1, keep_dims=True)
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_sent_dist, ~sent_sent_mask), [num_sent, -1])
    sent_only_loss = tf.clip_by_value(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = tf.reduce_mean(tf.nn.top_k(sent_only_loss, k=args.num_neg_sample)[0])

    loss = im_loss * args.im_loss_factor + sent_loss + sent_only_loss * args.sent_only_loss_factor
    return loss


def recall_k(im_embeds, sent_embeds, im_labels, ks=None):
    """
        Compute recall at given ks.
    """
    sent_im_dist = pdist(sent_embeds, im_embeds)
    def retrieval_recall(dist, labels, k):
        # Use negative distance to find the index of
        # the smallest k elements in each row.
        pred = tf.nn.top_k(-dist, k=k)[1]
        # Create a boolean mask for each column (k value) in pred,
        # s.t. mask[i][j] is 1 iff pred[i][k] = j.
        pred_k_mask = lambda topk_idx: tf.one_hot(topk_idx, labels.shape[1],
                            on_value=True, off_value=False, dtype=tf.bool)
        # Create a boolean mask for the predicted indicies
        # by taking logical or of boolean masks for each column,
        # s.t. mask[i][j] is 1 iff j is in pred[i].
        pred_mask = tf.reduce_any(tf.map_fn(
                pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)
        # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
        matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_max(matched, axis=1))
    return tf.concat(
        [tf.map_fn(lambda k: retrieval_recall(tf.transpose(sent_im_dist), tf.transpose(im_labels), k),
                   ks, dtype=tf.float32),
         tf.map_fn(lambda k: retrieval_recall(sent_im_dist, im_labels, k),
                   ks, dtype=tf.float32)],
        axis=0), sent_im_dist

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.

    function copied from - https://stackoverflow.com/questions/41273361/get-the-last-output-of-a-dynamic-rnn-in-tensorflow
    """
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    
    return res

def weight_l2_regularizer(initial_weights, scale, scope=None):
  """Returns a function that can be used to apply L2 regularization to weights.
  Small values of L2 can help prevent overfitting the training data.
  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.
  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.
  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(weights):
    """Applies l2 regularization to weights."""
    with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      weight_diff = initial_weights - weights
      return standard_ops.multiply(my_scale, nn.l2_loss(weight_diff), name=name)

  return l2

def setup_initialize_fc_layers(feats, parameters, scope_in, train_phase, args):
    for i, params in enumerate(parameters):
            scaling = params['scaling']
            outdim = len(scaling)
            cca_mean, cca_proj = params[scope_in + '_mean'], params[scope_in + '_proj']
            weights_init = tf.constant_initializer(cca_proj, dtype=tf.float32)
            weight_reg = weight_l2_regularizer(params[scope_in + '_proj'], args.cca_weight_reg)
            if (i + 1) < len(parameters):
                activation_fn = tf.nn.relu
            else:
                activation_fn = None

            feats = fully_connected(feats - cca_mean, outdim, activation_fn=activation_fn,
                                    weights_initializer = weights_init,
                                    weights_regularizer = weight_reg,
                                    scope = scope_in + '_embed_' + str(i)) * scaling

    feats = tf.nn.l2_normalize(feats, 1, epsilon=1e-10)
    return feats

def setup_sent_model(tokens, train_phase, vecs, max_length, args, fc_dim = 2048, embed_dim = 512):
    word_embeddings = tf.get_variable('word_embeddings', vecs.shape, initializer=tf.constant_initializer(vecs))
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tokens)
    embed_l2reg = tf.nn.l2_loss(word_embeddings - vecs)

    if args.language_model == 'gru':
        source_sequence_length = tf.reduce_sum(tf.cast(tokens > 0, tf.int32), 1)
        encoder_cell = tf.nn.rnn_cell.GRUCell(fc_dim)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, embedded_words, dtype=tf.float32,
            sequence_length=source_sequence_length)

        final_outputs = extract_axis_1(encoder_outputs, source_sequence_length-1)
        outputs = fully_connected(final_outputs, embed_dim, activation_fn = None,
                                  weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                  scope = 'phrase_encoder')

        s_embed = tf.nn.l2_normalize(outputs, 1, epsilon=1e-10)
    else:
        num_words = tf.reduce_sum(tf.to_float(tokens > 0), 1, keep_dims=True) + 1e-10
        average_word_embedding = tf.nn.l2_normalize(tf.reduce_sum(embedded_words, 1) / num_words, 1)
        if args.language_model == 'attend':
            context_vector = tf.tile(tf.expand_dims(average_word_embedding, 1), (1, max_length, 1))
            attention_inputs = tf.concat((context_vector, embedded_words), 2)
            attention_weights = fully_connected(attention_inputs, 1, 
                                                weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
                                                scope='word_attention')
            
            attention_weights = tf.expand_dims(tf.nn.softmax(tf.squeeze(attention_weights)), 2)
            sent_feats = tf.reshape(tf.nn.l2_normalize(tf.reduce_sum(embedded_words * attention_weights, 1), 1), [-1, vecs.shape[1]])
        else:
            sent_feats = average_word_embedding

        if args.init_filename:
            parameters = pickle.load(open(args.init_filename, 'rb'))
            s_embed = setup_initialize_fc_layers(sent_feats, parameters, 'lang', train_phase, args)
        else:
            sent_fc1 = add_fc(sent_feats, fc_dim, train_phase,'sent_embed_1')
            sent_fc2 = fully_connected(sent_fc1, embed_dim, activation_fn=None,
                                       weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
                                       scope = 'sent_embed_2')
            s_embed = tf.nn.l2_normalize(sent_fc2, 1, epsilon=1e-10)

    return s_embed, embed_l2reg

def setup_img_model(im_feats, train_phase, args, fc_dim = 2048, embed_dim = 512):
    if args.init_filename:
        parameters = pickle.load(open(args.init_filename, 'rb'))
        i_embed = setup_initialize_fc_layers(im_feats, parameters, 'vis', train_phase, args)
    else:
        im_fc1 = add_fc(im_feats, fc_dim, train_phase, 'im_embed_1')
        im_fc2 = fully_connected(im_fc1, embed_dim, activation_fn=None,
                                 weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
                                 scope = 'im_embed_2')
        i_embed = tf.nn.l2_normalize(im_fc2, 1, epsilon=1e-10)

    return i_embed

def embedding_model(im_feats, tokens, train_phase, im_labels , vecs, 
                    max_length, args, fc_dim = 2048, embed_dim = 512):
    """
        Build two-branch embedding networks.
        fc_dim: the output dimension of the first fc layer.
        embed_dim: the output dimension of the second fc layer, i.e.
                   embedding space dimension.
    """
    # Image branch.
    i_embed = setup_img_model(im_feats, train_phase, args, fc_dim, embed_dim)

    # Text branch.
    s_embed, embed_l2reg = setup_sent_model(tokens, train_phase, vecs, max_length, args, fc_dim, embed_dim)
    return i_embed, s_embed, embed_l2reg

def setup_train_model(im_feats, sent_feats, train_phase, im_labels, vecs, max_length, args):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be True.)
    # im_labels 5b x b
    i_embed, s_embed, embed_l2reg = embedding_model(im_feats, sent_feats, train_phase, im_labels, vecs, max_length, args)
    loss = embedding_loss(i_embed, s_embed, im_labels, args) + args.word_embedding_reg * embed_l2reg
    return loss


