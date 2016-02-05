# builtins
import asyncio
import asyncio.subprocess as aiosubprocess
import aiohttp
from aiohttp import web
import inspect
import os
import re
import sys
import warnings
from collections import namedtuple, deque, defaultdict, \
    MutableSequence
from operator import itemgetter
from subprocess import Popen, PIPE
from typing import List

from decorator import contextmanager

# pip


try:
    # MUST BE IMPORTED BEFORE BOKEH
    from tornado.platform.asyncio import AsyncIOMainLoop, tornado
    AsyncIOMainLoop().install()
    # MUST BE IMPORTED BEFORE BOKEH
except AssertionError:
    warnings.warn('Could not swap out tornado AIO-loop')

from tornado.websocket import websocket_connect
from bokeh.models import ColumnDataSource, Range1d, FactorRange, HoverTool
from bokeh.plotting import figure, hplot, vplot
from bokeh.client import push_session
from bokeh.io import curdoc
from bokeh.palettes import RdYlGn5, RdYlGn7


class AsyncProcWrapper:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._p = None

    async def start(self):
        self._p = await asyncio.create_subprocess_exec(*self.args,
                                                       **self.kwargs)

    @property
    def p(self) -> aiosubprocess.Process:
        return self._p

    @property
    def stdin(self) -> aiosubprocess.streams.StreamWriter:
        return self.p.stdin

    @property
    def stdout(self) -> aiosubprocess.streams.StreamReader:
        return self.p.stdout

    @property
    def stderr(self) -> aiosubprocess.streams.StreamReader:
        return self.p.stderr

    @classmethod
    async def create(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        await obj.start()
        return obj

    async def communicate(self) -> List[bytes]:
        err = await self.stderr.read()
        out = await self.stdout.read()
        return out, err


async def async_subprocess(*args, **kwargs):
    return await AsyncProcWrapper.create(*args, **kwargs)


def make_ssh_subprocess(*tunnel_args):
    async def async_ssh_subprocess(*args, **kwargs):
        return await async_subprocess('ssh', *tunnel_args, '-tt',
                               '{0}'.format(' '.join(args)), **kwargs)

    def ssh_subprocess(*args, **kwargs):
        return Popen(['ssh', *tunnel_args, '-tt',
                               '{0}'.format(' '.join(args[0]))], **kwargs)
    return async_ssh_subprocess, ssh_subprocess


class BokehConsole:
    def __init__(self, n=10, max_line_len=200, input_bottom=True):
        self.n = n
        self.max_line_len = max_line_len
        self.source = self.make_source()
        self.p = self.make_plot()
        self.line_buffer = deque(self.source.data['text'])
        self._rotate = 1 if input_bottom else -1
        self._pos = 0 if input_bottom else -1
        super(BokehConsole, self).__init__()

    def make_source(self):
        return ColumnDataSource({'text': [''] * self.n,
                                 'zeros': [0] * self.n,
                                 'line': list(range(self.n))})

    def make_plot(self):
        p = figure(y_range=(Range1d(-1, self.n + 1)),
                   x_range=(Range1d(-2, self.max_line_len + 1)), tools='hover',
                   width=int(self.max_line_len * 6.35 + 160),
                   height=(self.n + 2) * 16 + 100)
        p.text('zeros', 'line', 'text', source=self.source)
        p.axis.visible = None
        p.toolbar_location = 'below'
        g = p.grid
        g.grid_line_color = '#FFFFFF'
        return p

    def _push_line(self, line):
        self.line_buffer.rotate(self._rotate)
        self.line_buffer[self._pos] = line
        self.source.data['text'] = list(self.line_buffer)

    def output_text(self, s):
        for line in s.split('\n'):
            if len(line) <= self.max_line_len:
                self._push_line(line)
            else:
                tokens = list()
                i = -1
                for token in line.split(' '):
                    i += 1 + len(token)
                    if i > self.max_line_len:
                        self._push_line(' '.join(tokens))
                        tokens = [token]
                        i = len(token)
                    else:
                        tokens.append(token)


FNULL = open(os.devnull, 'w')
JobProgress = namedtuple('JobProgress', 'name percent state')
Command = namedtuple('Command', 'command args kwargs')
# Unless explicitly defined as Nvidia device all GPUs are considered as cuda
# devices
CPU = namedtuple('CPU', 'dev load')
GPU = namedtuple('GPU', 'dev free nvdev')
GPUNv = namedtuple('GPUNv', 'nvdev free load')  # <- nvidia device
GPUComb = namedtuple('GPUComb', 'dev free load')

GPUProcess = namedtuple('GPUProcess', 'pid owner dev memusage')
GPUNvProcess = namedtuple('GPUNvProcess',
                          'nvdev pid memusage')  # <- nvidia device

nvgpu_used_tot_load = re.compile('(\d+)MiB / (\d+)MiB \|\W*(\d+)%')
nvgpu_nvdev = re.compile('\|\W*(\d+)[^\|]+\|\W*[\dA-F]+:[\dA-F]+:[\dA-F]+\.')
nvgpu_divider = re.compile('^\+(-+\+)+[\n\r]*$')

gpu_dev_nvdev_used_tot = re.compile(
    'Device  (\d).*nvidia-smi\W*(\d+).+ (\d+) of (\d+) MiB Used')

gpu_nvdev_pid_mem = re.compile('(\d+)\W*(\d+).+?(\d+)MiB \|[\n\r]*$')

cpu_dev_load = re.compile('%Cpu(\d+)\W*:.*?(\d+)\[')
cpu_dev_us_sy = re.compile('%Cpu(\d+)\W*:\W* (\d+\.\d+) us,\W+(\d+\.\d+)')


def nv_line2nvdev(line, prev_nvdev=None):
    if nvgpu_divider.match(line):
        return None

    res = nvgpu_nvdev.findall(line)
    return int(res[0]) if res else prev_nvdev


def nv_line2GPUNv(line, nvdev):
    res = nvgpu_used_tot_load.findall(line)
    if res:
        if nvdev is None:
            raise ValueError('Found a valid info line, but i have no device')
        used, tot, load = res[0]
        free = int(tot) - int(used)
        return GPUNv(nvdev, free, float(load))


def nv_line2GPUNvProcess(line):
    res = gpu_nvdev_pid_mem.findall(line)
    if not res:
        return None
    return GPUNvProcess(*(int(r) for r in res[0]))


def cuda_line2GPU(line) -> GPU:
    res = gpu_dev_nvdev_used_tot.findall(line)
    if not res:
        return None
    dev, nvdev, used, tot = res[0]
    free = int(tot) - int(used)
    return GPU(int(dev), free, int(nvdev))


def lines2CPUs(lines):
    if not isinstance(lines, str):
        lines = '\n'.join(lines)
    res = cpu_dev_load.findall(lines)
    if res:
        return [CPU(int(dev), float(load)) for dev, load in res]

    res = cpu_dev_us_sy.findall(lines)
    if res:
        return [CPU(int(dev), float(us) + float(sy)) for dev, us, sy in res]
    return []


class RessourceMonitor:
    def __init__(self, change_stream: asyncio.Queue,
                 async_exec=async_subprocess,
                 normal_exec=Popen):
        # self.sources = self.init_sources()
        self.async_exec = async_exec
        self.normal_exec = normal_exec
        self.change_stream = change_stream
        self.terminated = False

    def init_sources(self):
        _gpus = self.gpus_mem()
        _cpus = self.cpus()

        source = dict(cpu=ColumnDataSource({
            'cpu_dev': [gpu.dev for gpu in _cpus],
            'cpu_load': [gpu.load for gpu in _cpus]}),
            gpumem=ColumnDataSource({
                'gpu_dev': [gpu.dev for gpu in _gpus],
                'gpu_free': [gpu.free for gpu in _gpus],
            }),
            gpuclock=ColumnDataSource({
                'gpu_dev': [gpu.dev for gpu in _gpus],
                'gpu_load': [gpu.free for gpu in _gpus],
            }))
        return source

    @staticmethod
    def gpus_mem(subproc_exec=Popen) -> List[GPU]:
        _gpus = list()
        for line in subproc_exec(['cuda-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            gpu = cuda_line2GPU(line)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    async def gpus_mem_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('cuda-smi',
                               stdout=asyncio.subprocess.PIPE,
                               stderr=FNULL)

        data = await p.stdout.read()
        _gpus = list()
        for line in data.decode('utf8').split('\n'):
            gpu = cuda_line2GPU(line)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    def nv_gpus_mem_load(subproc_exec=Popen) -> List[GPUNv]:
        _gpus = list()
        nvdev = None
        for line in subproc_exec(['nvidia-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            nvdev = nv_line2nvdev(line, nvdev)
            gpu = nv_line2GPUNv(line, nvdev)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @staticmethod
    async def nv_gpus_mem_load_coro(subproc_exec=async_subprocess) -> List[
        GPUNv]:
        p = await subproc_exec(['nvidia-smi'],
                               stdout=PIPE)
        data = await p.stdout.read()
        _gpus = list()
        nvdev = None
        for line in data.decode('utf8').split('\n'):
            nvdev = nv_line2nvdev(line, nvdev)
            gpu = nv_line2GPUNv(line, nvdev)
            if gpu:
                _gpus.append(gpu)
        return _gpus

    @classmethod
    def nv2cuda(cls, subproc_exec=Popen):
        gpus = cls.gpus_mem(subproc_exec)
        return dict((gpu.nvdev, gpu.dev) for gpu in gpus)

    @classmethod
    async def nv2cuda_coro(cls, subproc_exec=async_subprocess):
        gpus = await cls.gpus_mem_coro(subproc_exec)
        return dict((gpu.nvdev, gpu.dev) for gpu in gpus)

    @classmethod
    def gpus_comb(cls, subproc_exec=Popen):
        nv2cuda = cls.nv2cuda(subproc_exec)
        nvgpus = cls.nv_gpus_mem_load(subproc_exec)
        return [GPUComb(nv2cuda[gpu.nvdev], *gpu[1:]) for gpu in nvgpus]

    @classmethod
    async def gpus_comb_coro(cls, subproc_exec=async_subprocess):
        nv2cuda, nvgpus = await asyncio.gather(cls.nv2cuda_coro(subproc_exec),
                                               cls.nv_gpus_mem_load_coro(
                                                   subproc_exec))

        return [GPUComb(nv2cuda[gpu.nvdev], *gpu[1:]) for gpu in nvgpus]

    @classmethod
    def gpu_procs(cls, subproc_exec=Popen):
        nv2cuda = cls.nv2cuda(subproc_exec)
        nvprocs = dict()
        for line in subproc_exec(['nvidia-smi'],
                                 stdout=PIPE).stdout.read().decode(
            'utf8').split('\n')[:-1]:
            proc = nv_line2GPUNvProcess(line)
            if proc:
                nvprocs[proc.pid] = proc

        procs = list()
        for nvproc, owner in zip(nvprocs.values(),
                                 cls.find_pid_owner(*nvprocs.keys(),
                                                    subproc_exec=subproc_exec)):
            procs.append(GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                                    nvproc.memusage))
        return procs

    @classmethod
    async def gpu_procs_coro(cls, subproc_exec=async_subprocess):
        nv2cuda, p = await asyncio.gather(cls.nv2cuda_coro(subproc_exec),
                                          subproc_exec(['nvidia-smi'],
                                                       stdout=PIPE))
        data = await p.stdout.read()
        nvprocs = dict()
        for line in data.decode(
                'utf8').split('\n')[:-1]:

            proc = nv_line2GPUNvProcess(line)
            if proc:
                nvprocs[proc.pid] = proc

        procs = list()
        owners = await cls.find_pid_owner_coro(nvprocs.keys(), subproc_exec)
        for nvproc, owner in zip(nvprocs.values(), owners):
            procs.append(GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                                    nvproc.memusage))
        return procs

    async def _nvproc2proc(self, subproc_exec, nvproc: GPUNvProcess,
                           pid2owner: dict, nv2cuda: dict, gpu_nvprocs: dict):
        if nvproc.pid not in pid2owner:
            owner = await self.find_pid_owner_coro(nvproc.pid,
                                                   subproc_exec=subproc_exec)
            owner = owner[0]
            pid2owner[nvproc.pid] = owner
            prev_proc = None
        else:
            owner = pid2owner[nvproc.pid]
            prev_proc = gpu_nvprocs.get(nvproc.pid, None)

        if prev_proc != nvproc:
            gpu_nvprocs[nvproc.pid] = nvproc
            proc = GPUProcess(nvproc.pid, owner, nv2cuda[nvproc.nvdev],
                              nvproc.memusage)
            await self.change_stream.put(proc)

    async def gpus_mon(self, loop: asyncio.BaseEventLoop = None,
                       ignore=tuple()):
        subproc_exec = self.async_exec
        nv2cuda, pid2owner = await asyncio.gather(
            self.nv2cuda_coro(subproc_exec),
            self.pid2owner_coro(subproc_exec))

        p = await self.async_exec('nvidia-smi', '-l', '1',
                                  stdout=asyncio.subprocess.PIPE,
                                  stderr=FNULL)

        loop = loop or asyncio.get_event_loop()
        gpus = dict()
        gpu_nvprocs = dict()
        do_GPUComb = GPUComb not in ignore
        do_GPUProcess = GPUProcess not in ignore
        nvdev = None
        tasks = list()
        seen_pids = list()
        while not self.terminated:
            line = await p.stdout.readline()
            line = line.decode()
            if do_GPUComb:
                nvdev = nv_line2nvdev(line, nvdev)
                nvgpu = nv_line2GPUNv(line, nvdev)

                # a gpu was found in stdout
                if nvgpu:
                    prev_gpu = gpus.get(nvgpu.nvdev, None)
                    # has anything changed?
                    if prev_gpu != nvgpu[1:]:
                        # translate to cuda dev and update gpus
                        gpu = GPUComb(nv2cuda[nvgpu.nvdev], *nvgpu[1:])
                        gpus[nvgpu.nvdev] = nvgpu[1:]

                        # put into change stream
                        await self.change_stream.put(gpu)
                    continue

            if do_GPUProcess:
                nvproc = nv_line2GPUNvProcess(line)
                if nvproc:
                    seen_pids.append(nvproc.pid)
                    tasks.append(
                        loop.create_task(self._nvproc2proc(subproc_exec,
                                                           nvproc,
                                                           pid2owner,
                                                           nv2cuda,
                                                           gpu_nvprocs)))
                    continue

            if tasks:
                await asyncio.wait(tasks)
                tasks.clear()

                dead_pids = set(gpu_nvprocs.keys()).difference(seen_pids)
                for dead_proc in (gpu_nvprocs[pid] for pid in dead_pids):
                    await self.change_stream.put(GPUProcess(dead_proc.pid,
                                                            pid2owner[
                                                                dead_proc.pid],
                                                            nv2cuda[
                                                                dead_proc.nvdev],
                                                            0))
                    gpu_nvprocs.pop(dead_proc.pid)
                seen_pids.clear()

    @staticmethod
    def cpus(subproc_exec=Popen):
        p = subproc_exec(['top', '-n2', '-b', '-p0'], stdout=PIPE, stderr=FNULL)
        data = p.stdout.read()
        return lines2CPUs(data.decode())

    @staticmethod
    async def cpus_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('top', '-n2', '-b',
                               stdout=asyncio.subprocess.PIPE,
                               stderr=FNULL)
        data = await p.stdout.read()
        return lines2CPUs(data.decode())

    async def cpus_mon(self):
        subproc_exec = self.async_exec
        p = await subproc_exec('top', '-b', '-p0',
                               stdout=PIPE)
        cpus = dict()
        while not self.terminated:
            line = (await p.stdout.readline()).decode()
            cpu = lines2CPUs(line)
            if cpu:
                cpu = cpu[0]
                prev_load = cpus.get(cpu.dev, None)
                if prev_load != cpu.load:
                    cpus[cpu.dev] = cpu.load
                    await self.change_stream.put(cpu)

    @staticmethod
    def find_pid_owner(*pids, subproc_exec=Popen):
        pids = list(str(pid) for pid in pids)
        pid_cs = ','.join(pids)
        p = subproc_exec(['ps', '-p', pid_cs, '-o', 'pid,user', 'h'],
                         stdout=PIPE,
                         stderr=PIPE)
        data, err = p.communicate()
        if err and err != b'Connection to localhost closed.\r\n':
            raise IOError(err)
        pid2owner = dict(re.findall('\W*(\d+) (\w+)', data.decode()))
        return [pid2owner.get(pid, None) for pid in pids]

    @staticmethod
    async def find_pid_owner_coro(*pids, subproc_exec=async_subprocess):
        pids = list(str(pid) for pid in pids)
        pid_cs = ','.join(pids)
        p = await subproc_exec('ps', '-p', pid_cs, '-o', 'pid,user', 'h',
                               stdout=PIPE,
                               stderr=PIPE)

        data, err = await p.communicate()
        if err and err != b'Connection to localhost closed.\r\n':
            raise IOError(err)
        pid2owner = dict(re.findall('\W*(\d+) (\w+)', data.decode()))
        return [pid2owner.get(pid, None) for pid in pids]

    @staticmethod
    def pid2owner(subproc_exec=Popen):
        p = subproc_exec('ps', '-A', '-o', 'pid,user', 'h',
                         stdout=PIPE,
                         stderr=PIPE)
        data, err = p.communicate()
        if err and err != b'Connection to localhost closed.\r\n':
            raise IOError(err)
        pid2owner = dict((int(pid), owner) for pid, owner in
                         re.findall('\W*(\d+) (\w+)', data.decode()))
        return pid2owner

    @staticmethod
    async def pid2owner_coro(subproc_exec=async_subprocess):
        p = await subproc_exec('ps', '-A', '-o', 'pid,user', 'h',
                               stdout=PIPE,
                               stderr=PIPE)
        data, err = await p.communicate()
        if err and err != b'Connection to localhost closed.\r\n':
            raise IOError(err)
        pid2owner = dict((int(pid), owner) for pid, owner in
                         re.findall('\W*(\d+) (\w+)', data.decode()))
        return pid2owner


BytesStdOut = namedtuple('BytesStdOut', 'bytes')
TextStdOut = namedtuple('TextStdOut', 'text')
BytesStdErr = namedtuple('BytesStdErr', 'bytes')
TextStdErr = namedtuple('TextStdErr', 'text')


class ChangeStream(asyncio.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentinel2waiters = defaultdict(list)
        self.terminated = False

    class BytesRedirecter:
        def __init__(self, change_stream: asyncio.Queue, copy_out,
                     wrap_class=BytesStdOut):
            self.change_stream = change_stream
            self._copy_out = copy_out
            self.wrap_class = wrap_class
            self.put_bytes = True if hasattr(wrap_class, 'bytes') else False

        def write(self, b):
            is_bytes = isinstance(b, bytes)
            # no change needed
            if (is_bytes and self.put_bytes) or (
                        not is_bytes and not self.put_bytes):
                self.change_stream.put_nowait(self.wrap_class(b))

            # is bytes, but should be str
            elif is_bytes:
                self.change_stream.put_nowait(self.wrap_class(b.decode()))

            # is not bytes, but should be
            else:
                self.change_stream.put_nowait(self.wrap_class(b.encode()))

            if self._copy_out:
                self._copy_out.write(b)

        def flush(self):
            if self._copy_out:
                self._copy_out.flush()

    class TextRedirecter:
        def __init__(self, change_stream: asyncio.Queue, copy_out,
                     wrap_class=TextStdOut):
            self.change_stream = change_stream
            self._copy_out = copy_out
            self.wrap_class = wrap_class
            self.put_str = True if hasattr(wrap_class, 'text') else False

        def write(self, s):
            is_text = isinstance(s, str)
            # no change needed
            if (is_text and self.put_str) or (not is_text and not self.put_str):
                self.change_stream.put_nowait(self.wrap_class(s))

            # is bytes, but should be str
            elif not is_text:
                self.change_stream.put_nowait(self.wrap_class(s.decode()))

            # is not bytes, but should be
            else:
                self.change_stream.put_nowait(self.wrap_class(s.encode()))

            if self._copy_out:
                self._copy_out.write(s)

        def flush(self):
            if self._copy_out:
                self._copy_out.flush()

    @contextmanager
    def redirect_stdout(self, copy=False, out_type=str, **kwargs):
        if copy:
            copy_out = sys.stdout
        else:
            copy_out = None

        if out_type is str:
            redirecter = self.TextRedirecter(self, copy_out, **kwargs)
        else:
            redirecter = self.BytesRedirecter(self, copy_out, **kwargs)

        stdout, sys.stdout = sys.stdout, redirecter  # io.TextIOWrapper(self)
        yield
        sys.stdout = stdout

    @contextmanager
    def redirect_stderr(self, copy=False, out_type=str, **kwargs):
        if copy:
            copy_out = sys.stdout
        else:
            copy_out = None

        if out_type is str:
            if 'wrap_class' not in kwargs:
                kwargs['wrap_class'] = TextStdErr
            redirecter = self.TextRedirecter(self, copy_out, **kwargs)
        else:
            if 'wrap_class' not in kwargs:
                kwargs['wrap_class'] = BytesStdErr
            redirecter = self.BytesRedirecter(self, copy_out, **kwargs)

        stdout, sys.stderr = sys.stderr, redirecter  # io.TextIOWrapper(self)
        yield
        sys.stderr = stdout

    def register_waiter(self, waiter_coro):
        if inspect.isfunction(waiter_coro):
            waiter_coro = waiter_coro()
        sentinel = next(waiter_coro)
        self.sentinel2waiters[sentinel].append(waiter_coro)

    async def start(self):
        while not self.terminated:
            change = await self.get()
            #print(change)
            for sentinel, waiters in self.sentinel2waiters.items():
                if isinstance(change, sentinel):
                    dead_waiters = list()
                    for waiter in waiters:
                        try:
                            waiter.send(change)
                        except StopIteration:
                            dead_waiters.append(waiter)
                    for waiter in dead_waiters:
                        waiters.remove(waiter)


@contextmanager
def redirect_to_changestream(change_stream: asyncio.Queue, err=False):
    stdout = sys.stdout


class BokehPlots:
    def __init__(self, change_consumer: ChangeStream,
                 async_exec=async_subprocess,
                 normal_exec=Popen):
        self.async_exec = async_exec
        self.normal_exec = normal_exec
        self.change_consumer = change_consumer
        self.terminated = False

    @staticmethod
    def _drop_in(data, field, idx, val):
        data[field] = data[field][:idx] + [val] + data[field][idx + 1:]

    @staticmethod
    def _bar_source(names, *val_pairs):
        l = len(names)
        d = {
            'zeros': [0] * l,
            'name': [str(n) for n in names],
            'name_low': [str(n) + ':0.9' for n in names],
            'name_high': [str(n) + ':0.1' for n in names]
        }
        for val_name, values in val_pairs:
            if isinstance(values, (tuple, MutableSequence)):
                d[val_name] = values
            else:
                d[val_name] = [values] * l

        return d

    @staticmethod
    def _swapaxes(kwargs):
        kwargs['x_range'], kwargs['y_range'] = kwargs['y_range'], kwargs[
            'x_range']
        kwargs['x_axis_label'], kwargs['y_axis_label'] = kwargs['y_axis_label'], \
                                                         kwargs[
                                                             'x_axis_label']

    def cpu_bars(self, vertical=True, **kwargs):
        cpus = sorted(RessourceMonitor.cpus(subproc_exec=self.normal_exec),
                      key=itemgetter(0))
        devices, loads = tuple(zip(*cpus))
        source = ColumnDataSource(self._bar_source(devices, ('load', loads)))

        val_range = Range1d(0, 100)
        name_range = FactorRange(factors=source.data['name'])
        kwargs['x_range'] = name_range
        kwargs['y_range'] = val_range
        kwargs['x_axis_label'] = 'device'
        kwargs['y_axis_label'] = '%'

        if not vertical:
            self._swapaxes(kwargs)

        p = figure(**kwargs, tools='hover', title='CPU load')

        if vertical:
            p.quad(left='name_low', right='name_high', top='load',
                   bottom='zeros', source=source)
        else:
            p.quad(left='zeros', right='load', top='name_high',
                   bottom='name_low', source=source)

        @self.change_consumer.register_waiter
        def waiter():
            change = yield CPU
            while not self.terminated:
                loads = list(source.data['load'])
                loads[change.dev] = change.load
                source.data['load'] = loads
                change = yield

        return p

    def gpu_bars(self, vertical=True, **kwargs):
        gpus = sorted(RessourceMonitor.gpus_comb(subproc_exec=self.normal_exec),
                      key=itemgetter(0))

        devices, free_mems, loads = tuple(zip(*gpus))
        max_free = max(free_mems) * 1.5
        source = ColumnDataSource(self._bar_source(devices,
                                                   ('free', list(free_mems)),
                                                   ('load', list(loads))))

        def waiter():
            change = yield GPUComb
            while not self.terminated:
                self._drop_in(source.data, 'load', change.dev, change.load)
                self._drop_in(source.data, 'free', change.dev, change.free)
                change = yield

        self.change_consumer.register_waiter(waiter())

        name_range = FactorRange(factors=source.data['name'])

        def makefig(name_range, val_range, val_name, title, ylabel):
            kwargs['x_range'] = name_range
            kwargs['y_range'] = val_range
            kwargs['x_axis_label'] = 'device'
            kwargs['y_axis_label'] = ylabel

            if not vertical:
                self._swapaxes(kwargs)
            return figure(**kwargs, tools=[], title=title)

        p1, p2 = (makefig(name_range, Range1d(0, 100), 'load', 'GPU load', '%'),
                  makefig(name_range, Range1d(0, max_free), 'free',
                          'GPU free memory', 'MiB'))

        if vertical:
            p = vplot(p1, p2)
            p1.quad(left='name_low', right='name_high', top='load',
                    bottom='zeros', source=source)
            p2.quad(left='name_low', right='name_high', top='free',
                    bottom='zeros', source=source)
        else:
            p = hplot(p1, p2)
            p1.quad(left='zeros', right='load', top='name_high',
                    bottom='name_low', source=source)
            p2.quad(left='zeros', right='free', top='name_high',
                    bottom='name_low', source=source)
        p1.add_tools(HoverTool(tooltips=[('load', "@load")]))
        p2.add_tools(HoverTool(tooltips=[('free', "@free")]))
        return p

    def user_total_memusage(self, vertical=True, **kwargs):
        procs = sorted(RessourceMonitor.gpu_procs(subproc_exec=self.normal_exec),
                       key=itemgetter(0))
        users = defaultdict(dict)
        for proc in procs:
            users[proc.owner][proc.pid] = proc.memusage

        def user2data(_users):
            owners = list(_users.keys())
            memusage = [sum(_users[owner]) for owner in owners]
            return self._bar_source(owners, ('memusage', memusage))

        source = ColumnDataSource(user2data(users))

        val_range = Range1d(0, max(source.data['memusage']) * 1.5)
        name_range = FactorRange(factors=source.data['name'])
        kwargs['x_range'] = name_range
        kwargs['y_range'] = val_range
        kwargs['x_axis_label'] = 'username'
        kwargs['y_axis_label'] = 'MiB'

        if not vertical:
            self._swapaxes(kwargs)

        p = figure(**kwargs, tools=[], title='Memory usage',
                   )

        if vertical:
            p.quad(left='name_low', right='name_high', top='memusage',
                   bottom='zeros', source=source)
        else:
            p.quad(left='zeros', right='memusage', top='name_high',
                   bottom='name_low', source=source)

        @self.change_consumer.register_waiter
        def waiter():
            loop = asyncio.get_event_loop()
            change = yield GPUProcess
            print(change)
            while not self.terminated:
                users[change.owner][change.pid] = change.memusage

                # use "is" here to make sure it's the integer 0 which is a
                # sentinel for "dead"
                if change.memusage is 0:
                    print(change, 'marked as dead')
                    loop.call_later(10, users[change.owner].pop, change.pid)

                source.data.update(user2data(users))
                owners = list(users.keys())
                if name_range.factors != owners:
                    name_range.factors = owners
                change = yield

        return p

    def console(self, stdout=True, stderr=True, **kwargs):
        console_kwargs = dict(
            (key, kwargs[key]) for key in ['n', 'max_line_len', 'input_bottom']
            if key in kwargs)

        consoles = list()

        def waiter(sentinel, console: BokehConsole):
            change = yield sentinel
            if hasattr(sentinel, 'bytes'):
                while not self.terminated:
                    console.output_text(change.bytes.decode())
                    change = yield
            else:
                while not self.terminated:
                    console.output_text(change.text)
                    change = yield

        if stdout:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_waiter(waiter(BytesStdOut, c))
            self.change_consumer.register_waiter(waiter(TextStdOut, c))
            consoles.append(c)

        if stderr:
            c = BokehConsole(**console_kwargs)
            self.change_consumer.register_waiter(waiter(BytesStdErr, c))
            self.change_consumer.register_waiter(waiter(TextStdErr, c))
            consoles.append(c)

        if len(consoles) == 1:
            return consoles[0].p
        return hplot(*(c.p for c in consoles))

    def progress(self, job_names, **kwargs):
        job_name2idx = dict((name, i) for i, name in enumerate(job_names))
        state2color = dict(zip(['dead', 'queued', 'init', 'running-1',
                                'running-2', 'running-3',
                                'complete'], reversed(RdYlGn7)))
        source = ColumnDataSource(self._bar_source(job_names, ('percent', 0),
                                                   ('color', RdYlGn5[1])))
        p = figure(
            x_range=Range1d(0, 100), height=(100 + 30 * len(job_names)),
            width=400,
            y_range=FactorRange(factors=source.data['name']),
            tools=[HoverTool(tooltips=[("job", "@name"),
                                       ("percent", "@percent")])],
            x_axis_label='%', y_axis_label='jobname',
            title='Job progress'

        )

        p.quad(left='zeros', right='percent', top='name_high',
               bottom='name_low', fill_color='color', source=source)

        @self.change_consumer.register_waiter
        def waiter():
            change = yield JobProgress
            while not self.terminated:
                idx = job_name2idx.get(change.name, None)
                if None:
                    warnings.warn('unknown job!: "{}"'.format(change))
                self._drop_in(source.data, 'percent', idx, change.percent)

                if state2color[change.state] != source.data['color'][idx]:
                    self._drop_in(source.data, 'color', idx,
                                  state2color[change.state])
                change = yield

        return p

    def serve(self, host='localhost', port=5006, session_id='test'):
        url = 'http://' + host + ':' + str(port) + '/'
        self.session = push_session(curdoc(),
                                    session_id=session_id,
                                    url=url)
        #self.session.show()


def start_bokeh(mon_port, bokeh_port, plot_gen: BokehPlots, changes: ChangeStream):
    loop = asyncio.get_event_loop()

    async def test():
        await changes.put(Command('test', None, None))

    async def do_start():
        p_bokeh = await async_subprocess('bokeh', 'serve', '--port', str(bokeh_port),
                                         '--host', 'localhost:' + str(bokeh_port),
                                         '--host', 'localhost:' + str(mon_port),
                                         stderr=PIPE)

        while True:
            line = await p_bokeh.stderr.readline()
            print(line)
            if b'clients connected' in line:
                print('breaking')
                break
        plot_gen.serve(port=bokeh_port, session_id='monitor')
        await test()
        return p_bokeh

    try:
        plot_gen.serve(port=bokeh_port, session_id='monitor')
        loop.run_until_complete(test())
    except OSError:
        return loop.run_until_complete(do_start())


from bokeh.embed import autoload_server


def start_tornado(loop, bokeh_port, mon_port, scr):
    re_localhost = re.compile(b'http://localhost:' + str(bokeh_port).encode())
    re_wslocalhost = re.compile(b'wss?://localhost:' + str(bokeh_port).encode())

    class MainHandler(tornado.web.RequestHandler):
        def get(self):
            print('starting a session', self.request)
            self.write(scr)

    class HTTPProxyHandler(tornado.web.RequestHandler):
        async def get(self):
            url = 'http://localhost:{0}{1}?{2}'.format(bokeh_port,
                                                    self.request.path,
                                                    self.request.query)
            print(url)
            in_resp = await aiohttp.get(url)
            print(in_resp.status)
            headers = dict(in_resp.headers)
            b_copy = bytearray()
            while not in_resp.content.at_eof():
                print('readline')
                b = await in_resp.content.read(in_resp.content._limit)
                b_copy += re_wslocalhost.sub(b'ws://localhost:' + str(mon_port).encode(), re_localhost.sub(b'', b))
            in_resp.close()
            headers['CONTENT-LENGTH'] = len(b_copy)

            for h, val in headers.items():
                self.set_header(h, val)

            self.prepare()
            self.write(bytes(b_copy))

    class WebSocketProxyHandler(tornado.websocket.WebSocketHandler):
        def initialize(self):
            self.conn = None

        async def await_conn(self, conn):
            self.conn = await conn

            async for msg in self.conn:
                self.write_message(msg.data)
                if self.conn is None:
                    break

        def open(self):
            conn = aiohttp.ws_connect('ws://localhost:{0}/ws?'.format(bokeh_port) + self.request.query)
            loop.create_task(self.await_conn(conn))

        def on_message(self, message):
            self.conn.send_str(message)

        def on_close(self):
            self.conn.close()
            self.conn = None
            print("WebSocket closed")

    application = tornado.web.Application([
            ("/", MainHandler),
            ("/ws", WebSocketProxyHandler),
            ("/.+", HTTPProxyHandler),
        ])
    application.listen(mon_port)
    print('starting proxy server: {0} -> {1}'.format(mon_port, bokeh_port))

"""

def main_server(loop, bokeh_port, mon_port, scr):
    re_localhost = re.compile(b'http://localhost:' + str(bokeh_port).encode())
    re_wslocalhost = re.compile(b'wss?://localhost:' + str(bokeh_port).encode())

    async def wsproxy(request):
        print('HERE', request)
        assert isinstance(request, web.Request)
        if 'path' not in request.match_info:
            return web.Response(body=scr)
        # params = '&'.join('='.join(item) for item in request.items())
        print('requested')
        url = 'http://localhost:{0}/{1}?{2}'.format(bokeh_port,
                                                    request.match_info['path'],
                                                    request.query_string)
        print(url)
        in_resp = await aiohttp.get(url)
        print('got response', in_resp.status)
        assert isinstance(in_resp, aiohttp.ClientResponse)
        print(in_resp.headers)
        headers = dict(in_resp.headers)
        # out_resp = web.StreamResponse(status=in_resp.status, headers=in_resp.headers)
        print('preparing')
        # await out_resp.prepare(request)
        b_copy = bytearray()
        while not in_resp.content.at_eof():
            print('readline')
            b = await in_resp.content.read(in_resp.content._limit)
            b_copy += re_wslocalhost.sub(b'ws://', re_localhost.sub(b'', b))

            # out_resp.write()
        in_resp.close()
        print('eof')
        headers['CONTENT-LENGTH'] = len(b_copy)
        return web.Response(body=bytes(b_copy), headers=headers)

    async def proxy(request):
        assert isinstance(request, web.Request)
        if 'path' not in request.match_info:
            return web.Response(body=scr)
        # params = '&'.join('='.join(item) for item in request.items())
        print('requested')
        url = 'http://localhost:{0}/{1}?{2}'.format(bokeh_port,
                                                    request.match_info['path'],
                                                    request.query_string)
        print(url)
        in_resp = await aiohttp.get(url)
        print('got response', in_resp.status)
        assert isinstance(in_resp, aiohttp.ClientResponse)
        print(in_resp.headers)
        headers = dict(in_resp.headers)
        # out_resp = web.StreamResponse(status=in_resp.status, headers=in_resp.headers)
        print('preparing')
        # await out_resp.prepare(request)
        b_copy = bytearray()
        while not in_resp.content.at_eof():
            print('readline')
            b = await in_resp.content.read(in_resp.content._limit)
            b_copy += re_wslocalhost.sub(b'ws://', re_localhost.sub(b'', b))

            # out_resp.write()
        in_resp.close()
        print('eof')
        headers['CONTENT-LENGTH'] = len(b_copy)
        return web.Response(body=bytes(b_copy), headers=headers)

        return out_resp

    app = web.Application()
    app.router.add_route('GET', '/ws{path:.*}', wsproxy)
    app.router.add_route('GET', '/{path:.*}', proxy)
    app.router.add_route('GET', '/', proxy)

    loop = asyncio.get_event_loop()
    handler = app.make_handler()
    print('creating server')
    return loop.create_server(handler, '0.0.0.0', mon_port)

"""

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Bokeh Application</title>
    </head>
    <body>
{script}
    </body>
</html>"""

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser('HPC monitor')
    parser.add_argument('-b', '--bokeh-port', type=int, default=5006,
                        dest='bp')
    parser.add_argument('-m', '--monitor-port', type=int, default=8080,
                        dest='mp')
    parser.add_argument('-t', '--tunnels', type=str, default=None, nargs='*',
                        help='specify tunnels to monitored hosts.'
                             'for special args encapsulate args to ssh with ""'
                             'If not specified just monitor this machine')
    args = parser.parse_args()
    execs = dict()
    if args.tunnels:
        for tunnel in args.tunnels:
            t_args = re.findall('([\'"]([^"\']+)["\'])|([^ ]+)', tunnel)
            t_args = [a[-1] if a[-1] else a[-1] for a in t_args]
            t_args = [a.strip(' ') for a in t_args]
            execs['_'.join(t_args)] = make_ssh_subprocess(*t_args)
    else:
        execs['localhost'] = async_subprocess, Popen


    _hplots = list()
    monitors = list()
    change_streams = list()
    for ex in execs.values():
        changes = ChangeStream()
        mon = RessourceMonitor(changes, async_exec=ex[0], normal_exec=ex[1])
        plot_gen = BokehPlots(changes, async_exec=ex[0], normal_exec=ex[1])
        _hplots.append(vplot(plot_gen.cpu_bars(False), plot_gen.gpu_bars(),
                             plot_gen.user_total_memusage()))
        monitors.append(mon)
        change_streams.append(changes)


    plots = hplot(*_hplots)
    bokeh_port = args.bp
    mon_port = args.mp
    p_bokeh = start_bokeh(mon_port, bokeh_port, plot_gen, changes)

    scr = autoload_server(plots, session_id=plot_gen.session.id)
    print(scr)
    scr = re.sub('http://localhost:{0}'.format(bokeh_port), '', scr)
    scr = TEMPLATE.format(script=scr).encode()

    loop = asyncio.get_event_loop()
    start_tornado(loop, bokeh_port, mon_port, scr)
    loop.run_until_complete(asyncio.gather(
        *(mon.gpus_mon(loop=loop) for mon in monitors),
        *(mon.cpus_mon() for mon in monitors),
        *(changes.start() for changes in change_streams)
    ))
