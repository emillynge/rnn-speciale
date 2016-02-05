import re
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import progressbar
import sys
from ipywidgets import (FloatProgress, FloatSlider, Text, Textarea, interact, interactive, fixed,
                        HBox)
from IPython.display import (clear_output, display, HTML)
from lasagneutils import Options
from collections import OrderedDict, deque
from bokeh.models import ColumnDataSource, Line
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

class SelfUpdatePlot(object):
    def __init__(self, title, x=None, y=None, x_updater=None):
        x = x or deque()
        y = y or deque()
        self.x_updater = x_updater

        self.source = ColumnDataSource(data=dict(x=x, y=y))
        self.p = figure(title=title, plot_height=300, plot_width=600, y_range=(0, 5))
        self.p.line('x', 'y', color="#2222aa", alpha=0.5, line_width=2, source=self.source, name="foo")
        show(self.p)

    def clear(self):
        self.source.data['x'].clear()
        self.source.data['y'].clear()
        self.push_notebook()

    def append(self, y, x=None):
        self.source.data['y'].append(y)
        if x is None:
            if self.x_updater:
                x = self.x_updater(x)
            elif self.source.data['x']:
                x = self.source.data['x'][-1] + 1
            else:
                x = 0

        self.source.data['x'].append(x)
        self.source.data['y'].append(y)
        self.push_notebook()

    def pop_left(self):
        if self.source.data['x']:
            self.source.data['x'].popleft()
            self.source.data['y'].popleft()

    def append_limited_len(self, y, x=None, l=200):
        if len(self.source.data['y']) >= l:
            self.pop_left()
        self.append(y, x)

    def push_notebook(self):
        self.source.push_notebook()



class ProgressBar(object):
    def __init__(self, max_val=None):
        self.max_val = max_val
        self.meter = None
        self.it = None

    def __len__(self):
        if self.max_val:
            return self.max_val
        if self.it:
            try:
                self.max_val = len(self.it)
            except:
                pass
        raise ValueError('len cannot be determined')

    @property
    def value(self):
        if not self.meter:
            raise ValueError('no meter, initialize first')

        return self.meter.value

    @value.setter
    def value(self, value):
        if not self.meter:
            raise ValueError('no meter, initialize first')

        if value > len(self):
            raise ValueError('{} is larger than {}'.format(value, len(self)))

        self.meter.value = value
        return

    def start(self):
        self.meter = FloatProgress(min=0, max=len(self))
        self.meter.value = 0
        display(self.meter)

    def __iter__(self):
        it = iter(self.it)

        if not self.meter:
            self.start()
        self.meter.value = 0

        while True:
            self.value += 1
            try:
                yield next(it)
            except StopIteration as e:
                self.value = len(self)
                return

    def __call__(self, iterable):
        self.it = iterable
        return self


class JupyterOptions(Options):
    def __init__(self, *args, **kwargs):
        d = OrderedDict(*args, **kwargs)
        super().__init__()
        self._i = interact(self.show_args, **d)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            for child in self._i.widget.children:
                if child._kwarg == key:
                    child.value = value

        super().__setitem__(key, value)



    def show_args(self, **kwargs):
        s = '<h3>Arguments:</h3><table>\n'
        for k,v in kwargs.items():
            s += '<tr><td>{0}</td><td>{1}</td></tr>\n'.format(k,v)
            self[k] = v
        s += '</table>'
        display(HTML(s))

    def display(self):
        self.show_args(**self)


class _ProgressPrinter(Textarea):
    strip = re.compile('(^[ \n]+)|([ \n]+$)')
    def __init__(self, startup_text='', max_lines=5):
        super().__init__(disabled=True)
        self._lines = deque()
        self.value = ''
        self.max_lines = max_lines
        self(startup_text)

    def __call__(self, new_text):
        if not new_text:
            return
        self._lines.append(new_text)
        if len(self._lines) > self.max_lines:
            self._lines.popleft()

        self.value = '\n'.join(self._lines)

    def write(self, s):
        self(self.strip.sub('', s))

    def flush(self):
        pass

    def __del__(self):
        super().__del__()


class ProgressPrinter(_ProgressPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        display(self)

    def __del__(self):
        self.visible = False
        super().__del__()


@contextmanager
def console_out(max_lines=5):
    perr = _ProgressPrinter(max_lines=max_lines)
    pout =_ProgressPrinter(max_lines=max_lines)
    console = HBox([pout, perr])
    display(console)
    with redirect_stdout(pout):
        with redirect_stderr(perr):
            yield None


class Bar(progressbar.Widget):
    __slots__ = ('ipywidget')

    def __init__(self):
        self.ipywidget = FloatProgress(min=0, max=100)
        self.ipywidget.value = 0
        display(self.ipywidget)

    def update(self, pb):
        self.ipywidget.value = int(pb.currval / float(pb.maxval) * 100) if pb.maxval else 0
        return ''


class Message(progressbar.Widget):
    'Returns progress as a count of the total (e.g.: "5 of 47")'

    __slots__ = ('message', 'fmt', 'max_width')

    def __init__(self, message, max_width=None):
        self.message = message
        self.max_width = max(max_width or len(message), 1)
        self.fmt = ' {:' + str(self.max_width) + '} '

    def update(self, pbar):
        return self.fmt.format(self.message[:self.max_width])

class ProgressBar2(progressbar.ProgressBar):
    def test__iter__(self):
        with console_out():
            self.fd = sys.stderr
            while True:
                yield next(self)


def pbar(what, max_val, *args):
    msg = Message(*args)
    return ProgressBar2(max_val, fd=sys.stderr,
                       widgets=[progressbar.SimpleProgress(), ' ',
                                what, msg, progressbar.Percentage(),
                                Bar()]), msg