import inspect
from time import time
import collections

__author__ = 'emil'
from collections import deque, UserDict, Sequence, MutableSequence
import codecs
import os
from operator import itemgetter
from io import StringIO
import numpy as np
from functools import wraps
from numpy import array
from threading import Lock
from _pyio import TextIOWrapper, TextIOBase, IncrementalNewlineDecoder
from _io import TextIOWrapper as _TextIOWrapper
from abc import ABCMeta, abstractmethod
import theano.tensor.basic
from elymetaclasses import SingleDispatchMetaClass
from multiprocessing import Queue, Process
from threading import Thread
from multiprocessing.queues import Empty
from io import TextIOBase as _TextIOBase
from io import StringIO as _StringIO
from io import BytesIO as _BytesIO
from time import sleep


class InputStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, size=-1) -> Sequence:
        pass

    @abstractmethod
    def close(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is InputStream:
            if any("read" in B.__dict__ for B in C.__mro__):
                return True
            return NotImplemented
InputStream.register(TextIOWrapper)


class OutputStream(metaclass=ABCMeta):
    @abstractmethod
    def write(self, s) -> int:
        pass

    @abstractmethod
    def close(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is OutputStream:
            if any("write" in B.__dict__ for B in C.__mro__):
                return True
            return NotImplemented
OutputStream.register(TextIOWrapper)


class IOStream(InputStream, OutputStream):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is IOStream:
            if (any("read" in B.__dict__ for B in C.__mro__) and
                    any("write" in B.__dict__ for B in C.__mro__)):
                return True
            return NotImplemented


def sub_seq4stream(buffer_sz=1000, n_chunks=-1, custom_concat=None):
    """
    Make a decorator that switches out first argument such that it takes a stream instead of a sequence
    Thus it is possible to swap out a very long sequence for a stream to minimize memory usage
    :param buffer_sz: size of sequence to keep in memory
    :param n_chunks: number times to yield a result
    :param custom_concat: a function that concatenates new buffer with the remainder returned by wrapped function
        The default is to concatenate using a + b
    :return: decorator
    """
    if custom_concat is None:
        def extend(a, b):
            return a + b
    else:
        extend = custom_concat

    def decorator(func):
        """
        decorate func such that it takes a stream as first argument
        :param func: must be a generator, StopIteration Exception should have a value corresponding to the part of the
            supplied sequernce that should be carried over to the next buffer
        :return:
        """
        @wraps(func)
        def wrapper(stream: InputStream, *args, **kwargs):
            i = 0
            do_yield = lambda: True if n_chunks < 0 else lambda: i < n_chunks
            buffer = stream.read(buffer_sz)
            gen = func(buffer, *args, **kwargs)
            while do_yield():
                try:
                    yield next(gen)
                    i += 1
                except StopIteration as e:
                    buffer = extend(e.value, stream.read(buffer_sz))
                    if len(buffer) > len(e.value):
                        gen = func(buffer, *args, **kwargs)
                    else:
                        return
        return wrapper
    return decorator


# noinspection PyRedeclaration
class BatchGenerator(metaclass=SingleDispatchMetaClass):
    def __init__(self):
        self.stream = None
        self.orig_pos_start, self.orig_pos_end = None, None
        self.real_end, self.real_start = None, None
        self.sweep = 0

    def __init__(self, stream: InputStream, start=None, end=None):
        self.__init__()
        self.stream = self.copy_stream(stream)
        if any([start, end, self.orig_pos_start, self.orig_pos_end]):
            self.mpatch_interval(start, end)

    def __init__(self, bg: object):
        self.__init__()
        self.stream = self.copy_stream(bg.stream)
        self.orig_pos_start, self.orig_pos_end = bg.orig_pos_start, bg.orig_pos_end
        self.real_end, self.real_start = bg.real_end, bg.real_start
        self.mpatch_interval_if_needed()

    def __init__(self, fname: str, real_start=0, real_end=None):
        self.__init__()
        self.stream = open(fname)
        self.orig_pos_end = self.stream.seek(0, 2)
        self.orig_pos_start = self.stream.seek(0)
        self.real_start, self.real_end = real_start, real_end
        self.mpatch_interval_if_needed()

    def split(self, sub_stream_propotions: list):
        l = self.stream.seek(0, 2) - self.stream.seek(0)
        total_prop = sum(sub_stream_propotions)
        normed_props = [0] + [prop/total_prop for prop in sub_stream_propotions]
        fence_posts = [sum(normed_props[:i]) for i in range(1, len(normed_props) + 1)]
        splits_idx = [int(l * fence_posts) for fence_posts in fence_posts]
        return [BatchGenerator(self.stream, start=i1, end=i2) for i1, i2 in zip(splits_idx[:-1], splits_idx[1:])]

    def batches(self, batch_generator, opt, mode='simple',  status_printer=print):
        msg_q = Queue()
        workerpool = self.BatchWorkerPool()

        if mode == 'simple':
            workerpool.add_worker(*self.make_worker(msg_q, batch_generator, opt, mp=False))

        if 'split' in mode:
            split_args = mode.split('-')[1:]
            if split_args[0] == 'equal':
                sub_bgs = self.split([1] * int(split_args[1]))
            elif split_args[0] == 'prop':
                props = [int(p) for p in split_args[1:]]
                sub_bgs = self.split(props)
            else:
                raise ValueError('split mode unknown')

            workerpool.add_workers(bg.make_worker(msg_q, batch_generator, opt, worker_name=i) for i, bg in enumerate(sub_bgs))

        workerpool.start_workers()
        it = iter(workerpool.iter_batch_q_forever())
        workerpool.rotate()
        next(it)
        for batch_q in it:
            batch = batch_q.get(timeout=.1)
            if batch is StopIteration:
                workerpool.terminate_curr()
            else:
                yield batch

            while not msg_q.empty():
                status_printer(msg_q.get())

        while not msg_q.empty():
            status_printer(msg_q.get())
        workerpool.close()

    class BatchWorkerPool(deque):
        Worker = collections.namedtuple('Worker', 'worker batch_queues')

        class Iter:
            def __init__(self, workerpool, return_func):
                self.return_func = return_func
                self.workerpool = workerpool

            def __iter__(self):
                return self

            def __next__(self):
                if not self.workerpool:
                    raise StopIteration()
                self.workerpool.rotate()
                return self.return_func()

        @property
        def curr_worker(self):
            return self[0].worker if self else None

        @property
        def curr_batch_q(self):
            return self[0].batch_queues if self else None

        def add_worker(self, worker, batch_q):
            self.append(self.Worker(worker, batch_q))

        def add_workers(self, gen):
            for worker, batch_q in gen:
                self.add_worker(worker, batch_q)

        def start_workers(self):
            for worker, _ in self:
                worker.start()

        def iter_batch_q_forever(self):
            return self.Iter(self, lambda: self.curr_batch_q)

        def terminate_curr(self):
            worker, batch_q = self.popleft()
            if not batch_q.empty():
                print('Warning!: batchq not empty!')
            worker.terminate()

        def close(self):
            while self:
                self.terminate_curr()

        def __del__(self):
            self.close()


    def make_worker(self, msg_q: Queue, batch_generator, opt: dict, worker_name="", mp=True):
        batch_q = Queue(maxsize=1)
        stream = self.stream
        import traceback
        import sys


        sig = inspect.signature(batch_generator)
        func_opts = dict((k, opt[k]) for k in sig.parameters.keys() if k in opt)


        def batch_worker():
            t_start = time()

            def log(msg):
                msg_q.put('BW {2} - {0:3.0f}:\t{1}'.format(time()-t_start, msg, worker_name))
            try:
                log('started...')
                if opt.get('sweeps', None) is None:
                    test = lambda i: True
                else:
                    sweeps = opt['sweeps']
                    test = lambda i: i <= sweeps
                sweep = 1
                while test(sweep):
                    log('started sweep {0}'.format(sweep))
                    stream.seek(0)
                    for i, (x, y) in enumerate(batch_generator(stream, **func_opts)):
                        batch_q.put((x, y, i, sweep))
                    sweep += 1
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                log('Exception ocurred: {0}'.format(e))
                log(''.join(traceback.format_tb(exc_traceback, 10)))
                batch_q.put(StopIteration)
                raise e

            log('shutting down')
            batch_q.put(StopIteration)
        if mp:
            p = Process(target=batch_worker)
        else:
            p = Thread(target=batch_worker)

        return p, batch_q

    def copy_stream(self, stream) -> InputStream:
        """
        :param stream: input stream with read, tell and seek methods
        :return: new stream with its own state
        """
        raise NotImplementedError('stream {!r} cannot be copied due to unknown type')

    def copy_stream(self, stream: _BytesIO):
        raise NotImplementedError()

    def copy_stream(self, stream: _StringIO):
        raise NotImplementedError()

    def copy_stream(self, stream: _TextIOBase):
        """
        copy a file like stream
        :param stream: file pointer
        :return:
        """
        if os.path.isfile(stream.name):
            if hasattr(stream, '_mpatch_tell'):
                pos = stream.tell()
                stream.seek(0) # relative to start
                self.orig_pos_start = stream._mpatch_tell()

                stream_len = stream._mpatch_seek(0, 2)
                stream.seek(0, 2)
                self.orig_pos_end = stream._mpatch_tell() # relative to end
                if self.orig_pos_end == stream_len:
                    self.orig_pos_end = None
                stream.seek(pos)

            return open(stream.name, stream.mode)

        raise NotImplementedError()

    def mpatch_interval(self, start, end):
        self._set_real_interval(start, end)
        self.mpatch_interval_if_needed()

    def mpatch_interval_if_needed(self):
        if self._interval_patch_needed():
            self._do_interval_patch(self.real_end, self.real_start)

    def _interval_patch_needed(self):
        if self.real_end is not None and self.real_end != self.orig_pos_end:
            return True

        if self.real_start is not None and self.real_start != self.orig_pos_start and self.real_start != 0:
            return True
        return False

    def _set_real_interval(self, start, end):
        real_start = 0
        if self.orig_pos_start:
            real_start += self.orig_pos_start

        if start:
            real_start += start

        if real_start is None:
            real_start = 0

        if end is None:
            if self.orig_pos_end is None:
                real_end = None
            else:
                real_end = self.orig_pos_end
        else:
            if self.orig_pos_end is None:
                real_end = end
            else:
                if self.orig_pos_start is None:
                    raise ValueError('Cannot determine interval')
                real_end = self.orig_pos_start + end
        self.real_end, self.real_start = real_end, real_start

    def _do_interval_patch(self, real_end, real_start):

        if not hasattr(self.stream, '_mpatch_tell'):
            setattr(self.stream, '_mpatch_tell', self.stream.tell)
            setattr(self.stream, '_mpatch_read', self.stream.read)
            setattr(self.stream, '_mpatch_seek', self.stream.seek)

        from io import UnsupportedOperation

        def seek(cookie, whence=0):
            if whence == 0:
                pos = self.stream._mpatch_seek(real_start + cookie)
            elif whence == 1:
                pos = self.stream._mpatch_seek(cookie, whence)
            elif whence == 2:
                if real_end is None:
                    pos = self.stream._mpatch_seek(cookie, whence)
                else:
                    pos = self.stream._mpatch_seek(real_end - cookie)

            if real_start <= pos <= real_end:
                pos = pos - real_start if real_start else pos
                return pos
            raise UnsupportedOperation("out of bounds")

        file_end = self.stream._mpatch_seek(0, 2)

        def read(size=-1):
            _size = size
            if size > -1:
                pos = self.stream._mpatch_tell()
                i = 0
                if pos > file_end:
                    while pos > file_end:
                        self.stream._mpatch_read(1)
                        i += 1
                        pos = self.stream._mpatch_tell() - i
                        if i > 100:
                            raise Exception('BOOOOO', pos)
                    self.stream._mpatch_seek(pos)

                nextpos = pos + size
                if real_end:
                    nextpos = min(real_end, nextpos)
                size = nextpos - pos
                if size <= 0:
                    return ""
                return self.stream._mpatch_read(size)

            elif real_end is not None:
                size = real_end - self.stream._mpatch_tell()
                return self.stream._mpatch_read(size)
            else:
                return self.stream._mpatch_read(size)

        def tell():
            if real_start:
                return self.stream._mpatch_tell() + real_start
            else:
                return self.stream._mpatch_tell()

        setattr(self.stream, 'seek', seek)
        setattr(self.stream, 'read', read)
        setattr(self.stream, 'tell', tell)
        self.stream.seek(0)

    def __del__(self):
        self.stream.close()


class CatchReturnFromYield:
    def __init__(self, gen):
        self._value = None
        self.gen = gen

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, exc):
        self._value = exc.value

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.gen)
        except StopIteration as e:
            self.value = e
            raise e



class CharMap(UserDict):
    max_i = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse = dict((i, c) for c, i in self.items())
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def resize(self, max_i):
        self.__class__.max_i = max_i
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.reverse[value] = key

    def __missing__(self, char):
        i = len(self)
        if self.max_i <= i:
            raise ValueError('No more room')
        self[char] = i
        return i

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) > 1:
                return ''.join(self(vec) for vec in inp)
            if inp.dtype == np.bool:
                return self.reverse[int(self.idx_vec[inp])]
            return ''.join(self.reverse[int(i)] for i in inp)

        if isinstance(inp, MutableSequence):
            return np.concatenate(tuple(self(c) for c in inp)).reshape((len(inp), self.max_i))

        if isinstance(inp, str):
            return self.vectors[self[inp]]

        if isinstance(inp, (np.int, np.int64)):
            return self.reverse[int(inp)]


class SingletonCharMap:
    curr_charmap = CharMap()

    def __getattr__(self, item):
        if item not in ['curr_charmap']:
            return getattr(self.curr_charmap, item)
        return super().__getattr__(item)

    @classmethod
    def update_from_dict(cls, char_map: UserDict):
        CharMap.max_i = len(char_map)
        cls.curr_charmap = CharMap(char_map)


def cast_stream(fp, element_cast=lambda c: c, container_cast=list):
    if not hasattr(fp, '__read'):
        setattr(fp, '__read', fp.read)

    def read(*args, **kwargs):
        return container_cast(element_cast(c) for c in fp.__read(*args, **kwargs))

    setattr(fp, 'read', read)


def iter_n(iterable, iterations):
    if iterations is None:
        yield from iterable
    else:
        for _, item in zip(range(iterations), iterable):
            yield item



def windowed_chars(buffer, win_sz, concat=list):
    window = deque()
    window.extend(buffer[:win_sz])
    for char in buffer[win_sz:]:
        yield concat(window), char
        window.popleft()
        window.append(char)

    return buffer[-win_sz:]


@wraps(windowed_chars)
def windowed_vectors(*args):
    for window, char in windowed_chars(*args):
        yield CharMap(window), CharMap(char)


@wraps(windowed_chars)
def windowed_xvector(*args):
    for window, char in windowed_chars(*args):
        yield CharMap(window), CharMap[char]



def sequences(buffer, seq_len, win_sz):
    gen = windowed_chars(buffer, win_sz)
    i = 0
    while True:
        x, y = zip(*iter_n(gen, seq_len))
        if len(y) != seq_len:
            consumed_buffer = i * seq_len
            return buffer[consumed_buffer:]
        yield np.array(x), np.array(y)
        i += 1


def batches(buffer, batch_sz=None, seq_len=None, win_sz=None, **kwargs):
    gen = sequences(buffer, seq_len, win_sz)
    i = 0
    consumed_buffer = 0
    while True:
        xy = tuple(zip(*iter_n(gen, batch_sz)))
        if not xy:
            break
        x, y = xy
        if len(x) != batch_sz:
            break

        yield np.array(x), np.array(y)
        i += 1
        consumed_buffer = i * seq_len * batch_sz
    return buffer[consumed_buffer:]



def batch_gen_sliding(fp, win_sz, buf_sz, batch_sz):
    gen = windowed_xvector(fp, win_sz, buf_sz)
    x = deque()
    y = deque()
    for i, (_x, _y) in zip(range(batch_sz), gen):
        x.append(_x)
        y.append(_y)

    while True:
        yield (np.concatenate(x, axis=-1).reshape((batch_sz, win_sz, CharMap.max_i)),
               np.array(y))
        x.popleft(), y.popleft()
        _x, _y = next(gen)  # may raise StopIteration
        x.append(_x), y.append(_y)


def batch_gen(fp, win_sz, buf_sz, batch_sz):
    gen = windowed_vectors(fp, win_sz, buf_sz)
    while True:
        x = np.zeros((batch_sz, win_sz, CharMap.max_i))
        y = np.zeros((batch_sz, CharMap.max_i))
        i = 0
        for i, (_x, _y) in zip(range(batch_sz), gen):
            x[i, :, :] = _x
            y[i, :] = _y

        if i != batch_sz - 1:
            yield x[:i+1, :, :], y[:i+1, :]
            break
        yield x, y



class StreamSplitter(object):
    def __init__(self, fp, ratio, stream_sz=None, lock=None):
        self.fp = fp
        self.stream_lock = lock or Lock()
        self.stream_sz = stream_sz or self.get_stream_sz()
        self.split = int(self.stream_sz * ratio)

    @property
    def stream1(self):
        return self.SubStreamWrapper(self.SubStream(self.fp, 0, self.split, self.stream_lock))

    @property
    def stream2(self):
        return self.SubStreamWrapper(self.SubStream(self.fp, self.split, self.stream_sz, self.stream_lock))

    def get_stream_sz(self):
        buf = '-'
        self.fp.seek(0)
        l = -1
        while buf:
            buf = self.fp.read(1000)
            l += len(buf)
        self.fp.seek(0)
        return l

    class SubStreamWrapper(TextIOWrapper):
        @property
        def errors(self):
            return None

        @property
        def encoding(self):
            return None

        def _get_decoder(self):
            self._decoder = self.Decoder(None, False)
            return self._decoder

        def _get_encoder(self):
            return self.Encoder()

        class Decoder(IncrementalNewlineDecoder):
            def getstate(self):
                buf, flag = super().getstate()
                return str(buf), flag

        class Encoder(codecs.IncrementalEncoder):
            def encode(self, input, final=False):
                return input

    class SubStream(object):
        def __init__(self, fp: TextIOBase, start, stop, lock: Lock, *args, **kwargs):
            self.fp, self.start, self.stop, self.lock = fp, start, stop, lock
            self.pos = 0
            self.fp.seek(self.start)
            if self.lock.locked():
                raise ValueError('Stream not released!')
            self.lock.acquire()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

        def flush(self):
            self.fp.flush()

        @property
        def closed(self):
            return self.fp.closed

        def seekable(self):
            return self.fp.seekable()

        def writable(self):
            return self.fp.writable()

        def readable(self):
            return self.fp.readable()

        def seek(self, pos, **kwargs):
            if pos + self.start > self.stop or pos < 0:
                raise ValueError("Out of bounds")
            self.pos = pos
            self.fp.seek(self.start + pos)

        def close(self, *args, **kwargs):
            try:
                self.lock.release()
            except Exception:
                pass

        def read(self, n=-1, **kwargs):
            if self.closed:
                raise BlockingIOError('Stream closed')

            left = self.stop - self.fp.tell()
            n = n if 0 <= n < left else left
            self.pos += n
            return self.fp.read(n)
        readinto = read


class TrainCVTestSplit(StreamSplitter):
    def __init__(self, fp, train_prop, cv_test_prop, test_prop, stream_sz=None, lock=None):
        self.fp = fp
        self.total_prop = train_prop + cv_test_prop + test_prop
        self.stream_lock = lock or Lock()
        self.stream_sz = stream_sz or self.get_stream_sz()
        self.train_start = 0
        self.cv_test_start = int((train_prop / self.total_prop) * self.stream_sz)
        self.test_start = int(((train_prop + cv_test_prop) / self.total_prop) * self.stream_sz)

    @property
    def stream_train(self):
        return self.SubStreamWrapper(self.SubStream(self.fp, 0, self.cv_test_start, self.stream_lock))

    @property
    def stream_cvtest(self):
        return self.SubStreamWrapper(self.SubStream(self.fp, self.cv_test_start, self.test_start, self.stream_lock))

    @property
    def stream_test(self):
        return self.SubStreamWrapper(self.SubStream(self.fp, self.test_start, self.stream_sz, self.stream_lock))