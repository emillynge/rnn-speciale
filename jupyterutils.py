from ipywidgets import (FloatProgress, FloatSlider, Text, Textarea, interact, interactive, fixed)
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


class ProgressPrinter(Textarea):
    def __init__(self, startup_text='', max_lines=5):
        super().__init__(disabled=True)
        self._lines = deque()
        self.value = ''
        self.max_lines = max_lines
        self(startup_text)
        display(self)

    def __call__(self, new_text):
        self._lines.append(new_text)
        if len(self._lines) > self.max_lines:
            self._lines.popleft()

        self.value = '\n'.join(self._lines)

    def __del__(self):
        self.visible = False
        super().__del__()