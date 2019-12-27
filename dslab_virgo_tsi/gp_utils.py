import logging
import time

import gpflow
import tensorflow as tf

from dslab_virgo_tsi.model_constants import GaussianProcessConstants as GPConsts
from dslab_virgo_tsi.status_utils import status


# Reference why wrapper is needed: https://github.com/tensorflow/tensorflow/issues/27120
def get_optimization_step():
    @tf.function(autograph=False)
    def _optimization_step(optimizer, model: gpflow.models.SVGP, batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = - model.elbo(*batch)
            grads = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return objective

    return _optimization_step


class SVGaussianProcess(object):

    @staticmethod
    def run_optimization(model, iterations, train_dataset, minibatch_size):
        """
        Utility function running the optimization

        :param model: gpflow model
        :param iterations: number of iterations
        :param train_dataset: tensorflow Dataset placeholder
        :param minibatch_size: size of mini batch
        """
        # Create an Adam Optimiser action
        logf = []
        train_it = iter(train_dataset.batch(minibatch_size))
        optimizer = tf.optimizers.Adam(learning_rate=GPConsts.LEARNING_RATE)
        step, elbo_step = None, None
        start = time.time()
        optimization_step = get_optimization_step()
        for step in range(iterations):
            elbo = - optimization_step(optimizer, model, next(train_it))

            if step % 10 == 0:
                elbo_step = elbo.numpy()
                if step % 1000 == 0:
                    end = time.time()
                    if step != 0:
                        percentage_merging = int(100 * step / iterations)
                        percentage_overall = int(30 + 50 * step / iterations)
                        status.update_progress("Merging at: " + str(percentage_merging) + " %", percentage_overall)
                        logging.info("Step:\t{:<30}ELBO:\t{:>10}\t{:>10} second remaining"
                                     .format(step, elbo_step, int((iterations - step) / 1000) * (end - start)))
                    else:
                        percentage_merging = int(100 * step / iterations)
                        percentage_overall = int(30 + 50 * step / iterations)
                        status.update_progress("Merging at: " + str(percentage_merging) + " %", percentage_overall)
                        logging.info("Step:\t{:<30}ELBO:\t{:>10}"
                                     .format(step, elbo_step))
                    start = time.time()

                logf.append((step, elbo_step))

        logging.info("Step:\t{:<30}ELBO:\t{:>10}".format(step, elbo_step))
        return logf
