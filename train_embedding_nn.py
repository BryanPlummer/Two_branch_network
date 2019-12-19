from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
import tensorflow as tf

from dataset_utils import DatasetLoader
from retrieval_model import setup_train_model

FLAGS = None

def main(_):
    # Load data.
    data_loader = DatasetLoader(FLAGS)

    vocab_filename = os.path.join(FLAGS.feat_path, FLAGS.dataset, 'vocab.pkl')
    word_embedding_filename = os.path.join('data', 'mt_grovle.txt')
    embedding_length = 300
    print('Loading vocab')
    vecs = data_loader.build_vocab(vocab_filename, word_embedding_filename, embedding_length)
    print('Loading complete')

    num_ims, im_feat_dim = data_loader.im_feat_shape
    num_sents, sent_feat_dim = data_loader.sent_feat_shape
    steps_per_epoch = num_sents // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.max_num_epoch

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, im_feat_dim])
    sent_feat_plh = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * FLAGS.sample_size, sent_feat_dim])
    label_plh = tf.placeholder(tf.bool, shape=[FLAGS.batch_size * FLAGS.sample_size, FLAGS.batch_size])
    train_phase_plh = tf.placeholder(tf.bool)

    # Setup training operation.
    loss = setup_train_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh, vecs, sent_feat_dim, FLAGS)

    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               steps_per_epoch, 0.794, staircase=True)
    optim = tf.train.AdamOptimizer(init_learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=True)
    save_directory = os.path.join(FLAGS.save_dir, FLAGS.dataset, FLAGS.name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    average_loss = RunningAverage()
    save_checkpoint = os.path.join(save_directory, 'two_branch_chpt')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.resume:
            print('restoring checkpoint', args.resume)
            saver.restore(sess, args.resume)
            print('done')

        for i in range(num_steps):
            if i % steps_per_epoch == 0:
                # shuffle the indices.
                data_loader.shuffle_inds()
            im_feats, sent_feats, labels = data_loader.get_batch(
                    i % steps_per_epoch, FLAGS.batch_size, FLAGS.sample_size)
            feed_dict = {
                    im_feat_plh : im_feats,
                    sent_feat_plh : sent_feats,
                    label_plh : labels,
                    train_phase_plh : True,
            }

            [_, loss_val] = sess.run([train_step, loss], feed_dict = feed_dict)
            average_loss.update(loss_val)
            if i % FLAGS.display_interval == 0:
                print('Epoch: %d Step: %d Loss: %f' % ((i // steps_per_epoch) + 1, i, average_loss.avg()))
            if i % steps_per_epoch == 0 and i > 0 or (i + 1) == num_steps:
                print('Saving checkpoint at step %d' % (i + 1))
                saver.save(sess, save_checkpoint, global_step = global_step)

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0.

    def update(self, val):
        self.value_sum += val
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--name', type=str, default='Two_Branch_Network', help='Name of experiment')
    parser.add_argument('--feat_path', type=str, default='data', help='Path to the cached features.')
    parser.add_argument('--dataset', type=str, default='flickr', help='Dataset we are training on.')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory for saving checkpoints.')
    parser.add_argument('--resume', type=str, help='Full path location of file to restore and resume training from')
    # Training parameters.
    parser.add_argument('--language_model', type=str, default='avg', help='Type of language model to use. Supported: avg, attend, gru')
    parser.add_argument('--init_filename', type=str, default='', help='Full file path of a files with weights to initialize the network used by avg and attend')
    parser.add_argument('--display_interval', type=int, default=500, help='Number of iterations before displaying loss.')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--max_num_epoch', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--word_embedding_reg', type=float, default=5e-5, help='Weight on the L2 regularization of the pretrained word embedding.')
    parser.add_argument('--cca_weight_reg', type=float, default=1., help='Weight on the L2 regularization of some preinitialized layer weights.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.1,
                        help='Factor multiplied with sent only loss. Set to 0 for no neighbor constraint.')

    FLAGS, unparsed = parser.parse_known_args()
    assert FLAGS.language_model in ['avg', 'attend', 'gru']
    assert FLAGS.dataset in ['flickr', 'coco']
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
