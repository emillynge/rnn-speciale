from datagen import windowed_chars, windowed_vectors, batch_gen, \
    batch_gen_sliding, CharMap, BatchGeneratorBuffer
import lasagne as L
from theano import tensor as T
import theano
from functools import partial
from progressbar import ProgressBar as PB
import sys
import numpy as np
from time import time
from collections import OrderedDict
from argparse import ArgumentParser
from gzip import GzipFile
import pickle
from itertools import chain
from datagen import batches
from multiprocessing import Queue, Process
from multiprocessing.queues import Empty
from progressbar import ProgressBar as _ProgressBar
from _pyio import BytesIO, TextIOBase, TextIOWrapper, BufferedIOBase, BufferedRandom, BufferedReader, \
    IncrementalNewlineDecoder
from threading import Lock
import codecs
from functools import wraps

class ProgressBar(_ProgressBar):
    @property
    def value(self):
        return self.currval

    @value.setter
    def value(self, value):
        self.update(value)


def make_model(opt):
    l_in = L.layers.InputLayer((None, opt.seq_len, opt.win_sz, opt.features))
    l = l_in
    for _ in range(opt.n_hid_lay):
        print(l.output_shape)
        l = L.layers.LSTMLayer(l, opt.n_hid_unit, grad_clipping=100, nonlinearity=L.nonlinearities.tanh)


    print(l.output_shape)
    l_shp = L.layers.ReshapeLayer(l, (-1,opt.n_hid_unit))
    print(l_shp.output_shape)
    l_out = L.layers.DenseLayer(l_shp, num_units=CharMap.max_i, W=L.init.Normal(), nonlinearity=L.nonlinearities.softmax)
    print(l_out.output_shape)


    network_output = L.layers.get_output(l_out)
    all_params = L.layers.get_all_params(l_out)

    target_values = T.tensor3('target_output') # batch_sz, seq_len, n_classes


    cost = T.nnet.categorical_crossentropy(network_output, target_values.reshape2xy_batch_chunk((-1, CharMap.max_i))).mean()
    updates = L.updates.adagrad(cost, all_params, .01)
    f_train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    f_compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
    f_probs = theano.function([l_in.input_var],network_output, allow_input_downcast=True)
    return l_out, f_train, f_compute_cost, f_probs, OrderedDict(opt)


class LasagneModel(object):
    model_generator = make_model
    batch_generator = batches

    def __init__(self, opt, aux=None,  model_generator=None, batch_generator=None):
        self.model_generator = model_generator or self.model_generator
        self.batch_generator = batch_generator or self.batch_generator
        self.l_out, self.f_train, self.f_cost, self.f_predict, self.opt = self.model_generator(opt)
        self.aux = OrderedDict(aux)if aux else OrderedDict()
        self.err_data_handler = lambda err: None
        self.pbar = ProgressBar
        if 'train_error' not in self.aux:
            self.aux['train_error'] = list()

        if 'test_error' not in self.aux:
            self.aux['test_error'] = list()

    def change_opts(self, opt=None, **opt_changes):
        opt = opt or self.opt
        opt = OrderedDict(opt)
        opt.update(**opt_changes)
        self.l_out, self.f_train, self.f_cost, self.f_predict, self.opt = self.model_generator(opt)

    def _iterate_func(self, batch_gen: BatchGeneratorBuffer, func,
                      update_interval=10, pbar_cls=None):
        try:
            for i, (x, y) in enumerate(batch_gen.batches(self.opt, pbar_cls=pbar_cls)):
                err = func(x, y)
                self.err_data_handler(err)
                if i % update_interval == 0:
                    print('{0} Error: {1:2.3f}'.format(batch_gen.c, float(err)))

        except KeyboardInterrupt:
            print('Recieved interrupt...')
            print('Batch worker terminated')

    @wraps(_iterate_func)
    def train(self, *args, **kwargs):
        self._iterate_func(self._train_func,  *args, **kwargs)

    @wraps(_iterate_func)
    def test(self, *args, **kwargs):
        sweeps = self.opt.sweeps
        self.opt.sweeps = 1
        try:
            self._iterate_func(self._test_func, *args, **kwargs)
        finally:
            self.opt.sweeps = sweeps

    def _test_func(self, x, y):
        err = self.f_cost(x, y)
        self.aux['test_error'].append(err)
        return err

    def _train_func(self, x, y):
        err = self.f_train(x, y)
        self.aux['train_error'].append(err)
        return err

    def save_model(self, file):
        data = dict(params=L.layers.get_all_param_values(self.l_out),
                    opt=list(self.opt.items()),
                    model_generator=self.model_generator,
                    batch_generator=self.batch_generator,
                    aux=self.aux)

        def do_save(d, fp):
            fp_gz = GzipFile(filename='model.gz', fileobj=fp, mode='wb')
            pickle.dump(d, fp_gz)

        if isinstance(file, str):
            with open(file, 'wb') as fp:
                do_save(data, fp)
        else:
            do_save(data, file)

    @classmethod
    def read_model(cls, file):
        def do_read(fp):
            fp_gz = GzipFile(filename='model.gz', fileobj=fp, mode='rb')
            d = pickle.load(fp_gz)
            return d

        if isinstance(file, str):
            with open(file, 'rb') as fp:
                data = do_read(fp)
        else:
            data = do_read(file)

        params = data.pop('params')
        opt = Options.make(data.pop('opt'))
        model = cls(opt, **data)
        L.layers.set_all_param_values(model.l_out, params)
        return model

    @classmethod
    def set_default_model_generator(cls, model_generator):
        cls.model_generator = model_generator

    @classmethod
    def set_default_batch_generator(cls, batch_generator):
        cls.batch_generator = batch_generator


class Options(OrderedDict):
    def __init__(self, *args, **kwargs):
        self._short_args = OrderedDict(h='help')
        self._argsparser = ArgumentParser()
        super().__init__(*args, **kwargs)


    @staticmethod
    def _getter(key, self):
        return self[key]

    @staticmethod
    def _setter(key, self, value):
        self[key] = value

    def find_short_arg(self, key):
        shortarg = key[0]
        i = 1
        while shortarg in self._short_args and i < len(key):
            shortarg += key[i]
        self._short_args[shortarg] = key
        return shortarg

    def __setitem__(self, key, value):
        if key not in self or not hasattr(self, key):
            if not isinstance(key, str):
                raise ValueError('option names must be of type string')
            setattr(self.__class__, key, property(partial(self._getter, key),
                                        partial(self._setter, key)))

            arg = partial(self._argsparser.add_argument, '-' + self.find_short_arg(key), '--' + key, dest=key)
            kwargs = dict()
            if isinstance(value, dict):
                kwargs.update(value)
                value = value.get('default', None)
            else:
                kwargs['default'] = value

            if 'type' not in kwargs and value is not None:
                kwargs['type'] = type(value)
            arg(**kwargs)
        super().__setitem__(key, value)

    def parseargs(self, *args):
        if len(args) < 1:
            args = None
        else:
            args = [str(arg) for arg in args]
        self._argsparser.parse_args(args=args, namespace=self)

    @classmethod
    def make(cls, *args, **kwargs):
        class CustomOptions(cls):
            pass

        return CustomOptions(*args, **kwargs)



