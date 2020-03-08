import numpy as np
import tensorflow as tf


### tf implementation ###
def compute_clipped_policy_loss(negative_log_prob_action, old_negative_log_prob_action, advantage_estimates, clip_range):
    # compute r_t(theta) from negative log action probabilities
    ratios = tf.math.exp(old_negative_log_prob_action - negative_log_prob_action)

    # compute unclipped objective
    obj_unclipped = tf.math.multiply(advantage_estimates, ratios)

    # compute clipped objective
    clipped_ratios = tf.clip_by_value(ratios, 1-clip_range, 1+clip_range)
    obj_clipped = tf.math.multiply(advantage_estimates, clipped_ratios)

    # take the min of clipped and unclipped objective
    obj = tf.math.minimum(obj_unclipped, obj_clipped)

    # take the expected value of the objective over the batch of samples
    pg_loss = -tf.math.reduce_mean(obj)

    # compute the clipping rate
    batch_size = tf.size(obj_clipped)
    clipfrac = tf.math.reduce_mean(tf.cast(tf.math.greater(obj_unclipped, obj_clipped), tf.float32))# / batch_size

    return pg_loss, clipfrac


def compute_clipped_policy_loss_original(negative_log_prob_action, old_negative_log_prob_action, advantage_estimates, clip_range):
    ratio = tf.exp(old_negative_log_prob_action - negative_log_prob_action)
    pg_losses = -advantage_estimates * ratio
    pg_losses2 = -advantage_estimates * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 +
                                                  clip_range)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

    clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                      clip_range), tf.float32))

    return pg_loss, clipfrac

### numpy implementation ###
def compute_clipped_policy_loss_numpy(negative_log_prob_action, old_negative_log_prob_action, advantage_estimates, clip_range):
    # compute r_t(theta) from negative log action probabilities
    ratios = np.exp(old_negative_log_prob_action - negative_log_prob_action)

    # compute unclipped objective
    obj_unclipped = np.multiply(advantage_estimates, ratios)

    # compute clipped objective
    clipped_ratios = np.clip(ratios, 1-clip_range, 1+clip_range)
    obj_clipped = np.multiply(advantage_estimates, clipped_ratios)

    # take the min of clipped and unclipped objective
    obj = np.min(obj_unclipped, obj_clipped)

    # take the expected value of the objective over the batch of samples
    pg_loss = -np.mean(obj)

    # compute the clipping rate
    batch_size = obj_clipped.shape[0]
    clipfrac = np.sum(obj_unclipped == obj_clipped) / batch_size

    return pg_loss, clipfrac