__author__ = 'emil'
from collections import deque, UserDict, Sequence, MutableSequence
import codecs
from operator import itemgetter
from io import StringIO
import numpy as np
from functools import wraps
from numpy import array
from threading import Lock
from _pyio import TextIOWrapper, TextIOBase, IncrementalNewlineDecoder


class _CharMap(UserDict):
    max_i = 32
    def __init__(self, *args, **kwargs):
        super(_CharMap, self).__init__(*args, **kwargs)
        self.reverse = dict((i, c) for c, i in self.items())
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def resize(self, max_i):
        self.__class__.max_i = max_i
        self.vectors = np.eye(self.max_i, dtype=np.bool)
        self.idx_vec = np.arange(self.max_i)

    def __missing__(self, char):
        i = len(self)
        if self.max_i <= i:
            raise ValueError('No more room')
        self[char] = i
        self.reverse[i] = char
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




def iter_n(iterable, iterations):
    if iterations is None:
        yield from iterable
    else:
        for _, item in zip(range(iterations), iterable):
            yield item




CharMap = _CharMap()

def windowed_chars(fp, win_sz, buf_sz):
    if isinstance(fp, str):
        fp = StringIO(fp)

    window = deque()
    window.extend(fp.read(win_sz))
    buffer = fp.read(buf_sz)
    while buffer:
        for char in buffer:
            yield window, char
            window.popleft()
            window.append(char)
        buffer = fp.read(buf_sz)





@wraps(windowed_chars)
def windowed_vectors(*args):
    for window, char in windowed_chars(*args):
        yield CharMap(window), CharMap(char)


@wraps(windowed_chars)
def windowed_xvector(*args):
    for window, char in windowed_chars(*args):
        yield CharMap(window), CharMap[char]


def sequences(seq_len, fp, win_sz, buf_sz):
    gen = windowed_xvector(fp, win_sz, buf_sz)
    while True:
        x, y = zip(*iter_n(gen, seq_len))
        if len(y) != seq_len:
            return
        yield np.concatenate(x).reshape((seq_len, win_sz, -1)), np.array(y)


def batches(batch_sz=None, seq_len=None, fp=None, win_sz=None, buf_sz=None, features=32, **kwargs):
    CharMap.resize(features)
    gen = sequences(seq_len, fp, win_sz, buf_sz)
    while True:
        x, y = zip(*iter_n(gen, batch_sz))
        if len(x) != batch_sz:
            return
        y = np.concatenate(y).reshape((batch_sz, seq_len))
        y = np.array([CharMap(_y) for _y in CharMap(y)]).reshape(batch_sz, seq_len, -1)
        yield np.concatenate(x).reshape((batch_sz, seq_len, win_sz, -1)), y


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