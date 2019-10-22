# -*- coding: utf-8 -*-
"""

Created on May 17, 2018

@author:  neerbek

Copied from http://deeplearning.net/tutorial/code/lstm.py
"""
from __future__ import print_function
import os
import six.moves.cPickle as pickle
import getopt
import sys

from collections import OrderedDict
# import sys
import time

import numpy
# os.environ["OMP_NUM_THREADS"] = "1"
# (setenv "OMP_NUM_THREADS" "1")
# os.chdir(os.path.join(os.getenv("HOME"), "jan/taboo/taboo-selective/functionality/lstm/"))
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
import mon

datasets = {}

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    retain = 0.5
    proj = tensor.switch(use_noise,
                         (state_before *  # noqa: W504
                          trng.binomial(state_before.shape,
                                        p=retain, n=1,
                                        dtype=state_before.dtype)),
                         state_before * retain)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options, wEmb=None):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    if wEmb is None:
        randn = numpy.random.rand(options['n_words'],
                                  options['dim_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX)
    else:
        if not isinstance(wEmb, numpy.ndarray):
            raise Exception("expected a numpy array")
        lenx, leny = wEmb.shape
        if (options['n_words'] != -1 and lenx != options['n_words']) or leny != options['dim_proj']:
            raise Exception("expected shape ({}, {}), got ({}, {})".format(options['n_words'], options['dim_proj'], lenx, leny))
        params['Wemb'] = wEmb.astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +  # noqa: W504
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options, randomSeed):
    trng = RandomStreams(randomSeed)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-10
    if pred.dtype == 'float32':
        off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()                    # count true's
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])  # 1 - avg(true's)

    return valid_err


def train_lstm(
        dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        patience=10,  # Number of epoch to wait before early stop if no progress
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=10,  # Display to stdout the training progress every N updates
        decay_c=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto=None,  # The best model will be saved there
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        valid_batch_size=64,  # The batch size used for validation/test set.
        dataset='imdb',
        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
        # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
        glovePath=None,
        valid_portion=0.05,  # fraction of train set to be used as validation/dev set
        runOnly=False,
        randomSeed=None
):

    # Model options
    model_options = locals().copy()
    # print("model options", model_options)

    wordEmbMap = None
    wEmb = None
    if glovePath != None:
        wordEmbMap = mon.loadEmbeddings(glovePath, n_words, dim_proj)
        wEmbArray = [[] for i in range(len(wordEmbMap))]
        for (w, wordEmb) in wordEmbMap.items():
            wEmbArray[wordEmb.number] = wordEmb.representation
        wEmb = numpy.array(wEmbArray)

    datasets["imdb"] = (imdb.load_data, imdb.prepare_data)
    monPath = os.path.join(os.getenv("HOME"), "jan/ProjectsData/phd/DLP/Monsanto/data/trees/20191015c")
    monData = mon.MonsantoData(os.path.join(monPath, "trees0.zip"), wordEmbMap)
    datasets["mon0"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees1.zip"), wordEmbMap)
    datasets["mon1"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees2.zip"), wordEmbMap)
    datasets["mon2"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees3.zip"), wordEmbMap)
    datasets["mon3"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees0.zip"), wordEmbMap, useTestTrees=True)
    datasets["mon0t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees1.zip"), wordEmbMap, useTestTrees=True)
    datasets["mon1t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees2.zip"), wordEmbMap, useTestTrees=True)
    datasets["mon2t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees3.zip"), wordEmbMap, useTestTrees=True)
    datasets["mon3t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees0.zip"), wordEmbMap, manualSensitive=True)
    datasets["mon0ms"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees1.zip"), wordEmbMap, manualSensitive=True)
    datasets["mon1ms"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees2.zip"), wordEmbMap, manualSensitive=True)
    datasets["mon2ms"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees3.zip"), wordEmbMap, manualSensitive=True)
    datasets["mon3ms"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees0.zip"), wordEmbMap, useTestTrees=True, manualSensitive=True)
    datasets["mon0ms_t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees1.zip"), wordEmbMap, useTestTrees=True, manualSensitive=True)
    datasets["mon1ms_t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees2.zip"), wordEmbMap, useTestTrees=True, manualSensitive=True)
    datasets["mon2ms_t"] = (monData.loadData, imdb.prepare_data)
    monData = mon.MonsantoData(os.path.join(monPath, "trees3.zip"), wordEmbMap, useTestTrees=True, manualSensitive=True)
    datasets["mon3ms_t"] = (monData.loadData, imdb.prepare_data)

    load_data, prepare_data = get_dataset(dataset)

    if not dataset.startswith("mon"):
        print('Loading data {}'.format(dataset))
    else:
        print('Loading data {} {}'.format(dataset, monPath))
    train, valid, test = load_data(n_words=n_words, valid_portion=valid_portion,
                                   maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, wEmb)

    if reload_model:
        print("loading model...", reload_model)
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options, randomSeed=randomSeed)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    time_in_validation = 0
    eidx = 0
    if not runOnly:
        try:
            for eidx in range(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                epoch_start_time = time.time()
                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t]for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = f_grad_shared(x, mask, y)
                    f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print('bad cost detected: ', cost)
                        raise Exception('bad cost detected: {}'.format(cost))

                    if numpy.mod(uidx, dispFreq) == 0:
                        print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                    if saveto and numpy.mod(uidx, saveFreq) == 0:
                        if best_p is None:
                            params = unzip(tparams)
                        else:
                            params = best_p
                        name = "{}_best.npz".format(saveto)
                        print('Saving best... ' + name)
                        numpy.savez(name, history_errs=history_errs, **params)
                        pickle.dump(model_options, open("{}_best_options.pkl".format(saveto), 'wb'), -1)
                        name = "{}_running.npz".format(saveto)
                        print('Saving current... ' + name)
                        params = unzip(tparams)
                        numpy.savez(name, history_errs=history_errs, **params)
                        pickle.dump(model_options, open("{}_running_options.pkl".format(saveto), 'wb'), -1)
                        print('Done')

                    if numpy.mod(uidx, validFreq) == 0:
                        val_start_time = time.time()

                        use_noise.set_value(0.)
                        # without idx's: 21s/epoch
                        # with idx's: 34.92s/epoch
                        # kf = get_minibatches_idx(len(train[0]), batch_size)
                        kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
                        kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
                        # train_err = pred_error(f_pred, prepare_data, train, kf)
                        train_err = -1
                        valid_err = pred_error(f_pred, prepare_data, valid,
                                               kf_valid)
                        test_err = pred_error(f_pred, prepare_data, test, kf_test)

                        history_errs.append([valid_err, test_err])

                        best_test_err = numpy.array(history_errs)[:, 1].min()
                        if best_p is None or test_err <= best_test_err:
                            # we use test_err because we store validation data there
                            best_p = unzip(tparams)
                            bad_counter = 0

                        val_end_time = time.time()
                        time_in_validation += val_end_time - val_start_time
                        print('Errors: Train {:.4f} Valid {:.4f}'.format(train_err, valid_err), 'Test (best) {:.4f} ({:.4f})'.format(test_err, best_test_err), "Time: {:.2f}".format(val_end_time - val_start_time))

                        # if (len(history_errs) > patience and
                        #     test_err >= numpy.array(history_errs)[:-patience,
                        #                                           1].min()):
                        #     bad_counter += 1
                        #     if bad_counter > patience:
                        #         print('Early Stop!')
                        #         estop = True
                        #         break

                epoch_end_time = time.time()
                print('Epoch done. Seen {} samples. Time {:.2f}'.format(n_samples, epoch_end_time - epoch_start_time))

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    # kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    train_err = -1
    # train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    best_test_err = test_err
    if len(history_errs) > 0:
        best_test_err = numpy.array(history_errs)[:, 1].min()
    print('Best model: Errors: Train {:.4f} Valid {:.4f}'.format(train_err, valid_err), 'Test (best) {:.4f} ({:.4f})'.format(test_err, best_test_err))
    # print('Train ', 'N/A', 'Valid ', valid_err, 'Test ', test_err)

    if saveto and not runOnly:
        params = unzip(tparams)
        name = "{}_final.npz".format(saveto)
        print('Saving final...', name)
        numpy.savez(name, history_errs=history_errs, **params)
        pickle.dump(model_options, open("{}_final_options.pkl".format(saveto), 'wb'), -1)
        print('Done')
    print("The code run for {} epochs, with {:.2f} sec/epochs".format(eidx + 1, (end_time - start_time) / (1. * (eidx + 1))))
    print("Training took {:.2f}s, hereof {:.2f} sec in validation".format(end_time - start_time, time_in_validation))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    def usage(exitCode=0):
        print('lstm.py requires arguments check code file for input')
        sys.exit(exitCode)

    dim_proj = 128
    max_epochs = 100
    dispFreq = 10           # Display to stdout the training progress every N updates
    lrate = 0.0001          # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words = 30000         # Vocabulary size
    optimizer = adadelta    # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    saveto = None           # The best model will be saved there
    validFreq = 370         # Compute the validation error after this number of update.
    saveFreq = 1000
    maxlen = None           # Sequence longer then this get ignored
    batch_size = 16         # The batch size during training.
    valid_batch_size = 500  # The batch size used for validation/test set.
    dataset = "imdb"
    # use_dropout           # if False slightly faster, but worst test error
    reload_model = None     # Path to a saved model we want to start from.
    test_size = -1         # If >0, we keep only this number of test example.
    glovePath = None
    valid_portion = 0.05
    runOnly = False
    # Set the random number generators' seeds for consistency
    # SEED = 3984754
    SEED = None

    # parse input
    argv = sys.argv[1:]  # first arg is filename
    # print("WARN using debug args!")
    # argv = "--max_epochs 100 --test_size 250 --valid_batch_size 250 --validFreq 100 --dataset=mon1 --dim_proj 100 --glovePath /home/jneerbek/jan/ProjectsData/GloveWordEmb --valid_portion 0.12".split()
    try:
        opts, args = getopt.getopt(argv, "h", ["help", "dim_proj=", "max_epochs=", "dispFreq=", "lrate=", "n_words=", "optimizer=", "saveto=", "validFreq=", "saveFreq=", "maxlen=", "batch_size=", "valid_batch_size=", "dataset=", "reload_model=", "test_size=", "glovePath=", "valid_portion=", "runOnly", "randomSeed="])
    except getopt.GetoptError:
        usage(exitCode=2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(exitCode=0)
        elif opt in ("--dim_proj"):
            dim_proj = int(arg)
        elif opt in ("--max_epochs"):
            max_epochs = int(arg)
        elif opt in ("--dispFreq"):
            dispFreq = int(arg)
        elif opt in ("--lrate"):
            rlate = float(arg)
        elif opt in ("--n_words"):
            n_words = int(arg)
        elif opt in ("--optimizer"):
            if arg == "adadelta":
                optimizer = adadelta
            elif arg == "RMSprop":
                optimizer = rmsprop
            elif arg == "SGD":
                optimizer = sgd
            else:
                raise Exception("unknown optimizer: " + arg)
        elif opt in ("--saveto"):
            saveto = arg
        elif opt in ("--validFreq"):
            validFreq = int(arg)
        elif opt in ("--saveFreq"):
            saveFreq = int(arg)
        elif opt in ("--maxlen"):
            maxlen = int(arg)
        elif opt in ("--batch_size"):
            batch_size = int(arg)
        elif opt in ("--valid_batch_size"):
            valid_batch_size = int(arg)
        elif opt in ("--dataset"):
            dataset = arg
        elif opt in ("--reload_model"):
            reload_model = arg
        elif opt in ("--test_size"):
            test_size = int(arg)
        elif opt in ("--glovePath"):
            glovePath = arg
        elif opt in ("--valid_portion"):
            valid_portion = float(arg)
        elif opt in ["--runOnly"]:
            runOnly = True
        elif opt in ["--randomSeed"]:
            SEED = int(arg)

    if SEED is None:
        numpy.random.seed()
        SEED = numpy.random.randint(1000000)
    print("randomSeed: {}".format(SEED))
    numpy.random.seed(SEED)
    # DEBUG
    encoder = 'lstm'
    patience = 10  # Number of epoch to wait before early stop if no progress
    decay_c = 0.  # Weight decay for the classifier applied to the U weights.
    noise_std = 0.
    use_dropout = True  # if False slightly faster, but worst test error

    train_lstm(
        dim_proj=dim_proj,
        max_epochs=max_epochs,
        dispFreq=dispFreq,
        lrate=lrate,
        n_words=n_words,
        optimizer=optimizer,
        saveto=saveto,
        validFreq=validFreq,
        saveFreq=saveFreq,
        maxlen=maxlen,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        dataset=dataset,
        reload_model=reload_model,
        test_size=test_size,
        glovePath=glovePath,
        valid_portion=valid_portion,
        runOnly=runOnly,
        randomSeed=SEED
    )

