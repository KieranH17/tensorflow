
import numpy as np
import tensorflow as tf

# mostly adapted from tensorflow tutorial on building a cnn

tf.logging.set_verbosity(tf.logging.INFO)


# features are list of state layers, labels are list of outcomes
def mp_ccn_fn(features, labels, mode):

    # input layer
    input_layer = tf.reshape(features, [-1, 8, 8, 20])

    # conv layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu
    )

    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # conv layers 2 and pooling layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[8, 8],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 2 * 2 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # generate predictions
        "classes": tf.argmax(input=logits, axis=1),
        # predict
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # configure training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval metrics
    eval_metrics_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)



