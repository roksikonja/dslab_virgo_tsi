import logging

import gpflow
import tensorflow as tf


@tf.function(autograph=False)
def optimization_step(optimizer, model: gpflow.models.SVGP, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(*batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


class SVGaussianProcess(object):

    @staticmethod
    def run_adam(model, iterations, train_dataset, minibatch_size):
        """
        Utility function running the Adam optimiser

        :param model: GPflow model
        :param iterations: number of iterations
        :param train_dataset: Tensorflow Dataset placeholder
        :param minibatch_size: size of mini batch
        """
        # Create an Adam Optimiser action
        logf = []
        train_it = iter(train_dataset.batch(minibatch_size))
        adam = tf.optimizers.Adam()
        for step in range(iterations):
            elbo = - optimization_step(adam, model, next(train_it))
            if step % 10 == 0:
                elbo_step = elbo.numpy()
                if step % 1000 == 0:
                    logging.info("Step:\t{:<30}ELBO:\t{:>10}".format(step, elbo_step))

                logf.append(elbo_step)
        return logf
