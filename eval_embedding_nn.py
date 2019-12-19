from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import os
import numpy as np
import tensorflow as tf

from dataset_utils import DatasetLoader
from retrieval_model import  setup_sent_model, setup_img_model, recall_k

FLAGS = None

def get_embed(placeholders, feat_plh, feats):
    saver = tf.train.Saver(save_relative_paths=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Restore latest checkpoint or the given MetaGraph.
        ckpt_path = FLAGS.resume
        if ckpt_path.endswith('.meta'):
            ckpt_path = ckpt_path.replace('.meta', '')

        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')
        # For testing and validation, there should be only one batch with index 0.
        feed_dict = {placeholders['feat'] : feats,
                     placeholders['train_phase'] : False,
        }
        [features] = sess.run([feat_plh], feed_dict = feed_dict)

    tf.reset_default_graph()
    return features

def get_embedded_images(im_feats):
    num_ims, im_feat_dim = im_feats.shape
    im_feat_plh = tf.placeholder(tf.float32, shape=[num_ims, im_feat_dim])
    train_phase_plh = tf.placeholder(tf.bool)
    placeholders = {
        'feat' : im_feat_plh,
        'train_phase' : train_phase_plh,
    }

    img_embed = setup_img_model(im_feat_plh, train_phase_plh, FLAGS)
    i_feats = get_embed(placeholders, img_embed, im_feats)
    return i_feats

def get_embedded_sentences(sentences, vecs):
    num_sents, sent_feat_dim = sentences.shape

    # some language models won't be able to pass through the entire dataset at
    # once if they're large so let's split the number of sentences in a batch
    num_sent_splits = 2
    num_sent_batch = int(np.ceil(num_sents / float(num_sent_splits)))
    s_feats = []
    for batch_id in range(num_sent_splits):
        sent_feat_plh = tf.placeholder(tf.int32, shape=[num_sent_batch, sent_feat_dim])
        train_phase_plh = tf.placeholder(tf.bool)
        placeholders = {
            'feat' : sent_feat_plh,
            'train_phase' : train_phase_plh,
        }

        sent_embed = setup_sent_model(sent_feat_plh, train_phase_plh, vecs, sent_feat_dim, FLAGS)
        sent_feats = np.zeros((num_sent_batch, sent_feat_dim), np.int32)
        sent_start = batch_id * num_sent_batch
        sent_end = min((batch_id + 1) * num_sent_batch, num_sents)
        num_sentences = sent_end - sent_start
        sent_feats[:num_sentences] = sentences[sent_start:sent_end]
        s_feats.append(get_embed(placeholders, sent_embed[0], sent_feats))

    s_feats = np.vstack(s_feats)[:num_sents]
    return s_feats

def main(_):
    # Load data.
    data_loader = DatasetLoader(FLAGS, split=FLAGS.split)
    vocab_filename = os.path.join(FLAGS.feat_path, FLAGS.dataset, 'vocab.pkl')

    print('Loading vocab')
    vecs = data_loader.build_vocab(vocab_filename)
    print('Loading complete')

    i_feats = get_embedded_images(data_loader.im_feats)
    s_feats = get_embedded_sentences(data_loader.sent_feats, vecs)

    num_ims, im_feat_dim = data_loader.im_feat_shape
    num_sents, sent_feat_dim = data_loader.sent_feat_shape

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[num_ims, 512])
    sent_feat_plh = tf.placeholder(tf.float32, shape=[num_sents, 512])
    label_plh = tf.placeholder(tf.bool, shape=[num_sents, num_ims])
    placeholders = {
        'im_feat' : im_feat_plh,
        'sent_feat' : sent_feat_plh,
        'label' : label_plh,
    }

    recall = recall_k(im_feat_plh, sent_feat_plh, label_plh, ks=tf.convert_to_tensor([1,5,10]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Restoring checkpoint', FLAGS.resume)
        saver = tf.train.import_meta_graph(FLAGS.resume + '.meta')
        saver.restore(sess, FLAGS.resume)
        print('Done')

        feed_dict = {
            placeholders['im_feat'] : i_feats,
            placeholders['sent_feat'] : s_feats,
            placeholders['label'] : data_loader.labels,
        }
        
        [recall_vals, dist] = sess.run(recall, feed_dict = feed_dict)

    im2sent = [round(r * 100, 1) for r in recall_vals[:3]]
    sent2im = [round(r * 100, 1) for r in recall_vals[3:]]
    mr = round(np.mean(recall_vals) * 100, 1)
    print('im2sent: {:.1f} {:.1f} {:.1f} sent2im: {:.1f} {:.1f} {:.1f} mr: {:.1f}'.format(
        im2sent[0], im2sent[1], im2sent[2], sent2im[0], sent2im[1], sent2im[2], mr))

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--feat_path', type=str, default='data', help='Path to the cached features.')
    parser.add_argument('--dataset', type=str, default='flickr', help='Dataset we are training on.')
    parser.add_argument('--split', type=str, help='Data split to test.')
    parser.add_argument('--resume', type=str, help='checkpoint to resume from.')
    # Training parameters.
    parser.add_argument('--language_model', type=str, default='avg', help='Type of language model to use. Supported: avg, attend, gru')
    parser.add_argument('--init_filename', type=str, default='', help='Full file path of a files with weights to initialize the network used by avg and attend')
    parser.add_argument('--cca_weight_reg', type=float, default=0., help='Weight on the L2 regularization of some preinitialized layer weights.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation.')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of positive pair to sample.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
