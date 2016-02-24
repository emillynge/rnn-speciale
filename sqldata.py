import os
import warnings
import zipfile
from collections import UserDict, OrderedDict
from functools import lru_cache
import numpy as np
import io
from lasagnecaterer.menu import empty_fridge
import postgresql

db = postgresql.open(
    'pq://{0}@localhost:5432/speciale'.format(os.environ['USER']))
opt_cols = set(c[0] for c in db.query(
    "SELECT column_name FROM information_schema.columns WHERE table_name='opts';"))
n_epochs = 50


@lru_cache()
def add_col(table, data_type, col_name):
    db.execute(
        "ALTER TABLE {0} ADD COLUMN {2} {1}".format(table, data_type, col_name))


@lru_cache()
def insert_cmd(table, cols, pkey='path'):
    cmd = "INSERT INTO {2} ({0}) VALUES ({1}) ON CONFLICT ({3}) DO UPDATE " \
          "SET ({0}) = ({1});"
    cmd = cmd.format(', '.join(cols),
                     ', '.join("${0}".format(i)
                               for i in range(1, len(cols) + 1)),
                     table,
                     pkey)
    return db.prepare(cmd)


class ArrayBytesConv(io.BytesIO):
    def __call__(self, arr, *args, **kwargs):
        self.seek(0)
        if isinstance(arr, np.ndarray):
            np.save(self, arr)
            self.truncate()
            return self.getvalue()
        else:
            self.write(arr)
            self.truncate()
            self.seek(0)
            try:
                return np.load(self)
            except OSError:
                warnings.warn('Failed to load array!')
                return None


arrconv = ArrayBytesConv()


def insert_opts(path, opt):
    insert_cmd('opts',
               ('path',) + tuple(opt.keys()))(path, *tuple(opt.values()))


def insert_cols(opt):
    for key, value in opt.items():
        if key not in opt_cols:
            if isinstance(value, (int, float)):
                add_col('opts', 'NUMERIC', key)
            elif isinstance(value, (str,)):
                add_col('opts', 'TEXT', key)
            else:
                raise ValueError(
                    'unknown data type {0} for opt column'.format(type(value)))
            opt_cols.add(key)


insert_errors = insert_cmd('error', ('path', 'train', 'test'))


def model2db(model, path):
    model.opt['folder'] = path.split('/')[0]
    if 'lstm' in model.opt['folder']:
        arch = 'LSTM'
    elif 'gru' in model.opt['folder']:
        arch = 'GRU'
    else:
        arch = 'VRNN'
    model.opt['arch'] = arch

    if any(model.opt.keys()) not in opt_cols:
        insert_cols(model.opt)
    insert_opts(path, model.opt)

    try:
        insert_errors(path,
                      arrconv(np.array(model.cook.box['train_error_hist'])),
                      arrconv(np.array(model.cook.box['val_error_hist'])))
    except KeyError as e:
        warnings.warn('{0} - {1}'.format(e, path))


def fullpath2path(path):
    folder, fname = os.path.split(path)
    last_folder = folder.split(os.path.sep)[-1]
    return last_folder + '/' + fname


def dump_folder_to_db(folder, force=False):
    models = list()
    print(folder)
    loaded = set(c[0].strip() for c in db.query('SELECT path FROM opts'))
    for file in os.scandir(folder):
        if file.name.endswith('.lfr') and 'basemodel.lfr' not in file.name:
            if file.stat().st_size > 0:
                path = fullpath2path(file.path)
                if not (force or path not in loaded):
                    continue
                try:
                    model = empty_fridge(file.path)
                    model2db(model, path)
                    loaded.add(path)
                    print(file.name)
                except zipfile.BadZipFile:
                    print('Corrupt zip', file.name)


class Where:
    def __init__(self):
        self.stack = list()

    def __getattr__(self, item):
        sq = self.SubQuery(item, self)
        self.stack.append(sq)
        return sq

    class SubQuery:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent
            self.operator = None
            self.value = None

        def set_cond(self, operator, value, quote=True):
            if isinstance(value, str) and quote:
                value = "'{}'".format(value)
            self.operator = operator
            self.value = value
            return self.parent

        def __eq__(self, other): return self.set_cond('=', other)

        def __lt__(self, other): return self.set_cond('<', other)

        def __le__(self, other): return self.set_cond('<=', other)

        def __gt__(self, other): return self.set_cond('>', other)

        def __ge__(self, other): return self.set_cond('>=', other)

        def __ne__(self, other): return self.set_cond('!=', other)

        def __contains__(self, item):
            conds = ', '.join("'{}'".format(it) if isinstance(it, str) else it
                              for it in item)
            return self.set_cond('IN', '({})'.format(conds), quote=False)

        def __str__(self):
            return self.name + ' ' + self.operator + ' ' + str(self.value)

        def __repr__(self):
            return self.__str__()

    def __str__(self):
        return ' '.join(str(clause) for clause in self.stack)

    def __repr__(self):
        return self.__str__()

    @property
    def AND(self):
        self.stack.append('AND')
        return self

    @property
    def OR(self):
        self.stack.append('OR')
        return self


def err_from_q(query, col):
    if not isinstance(col, str):
        col = '{}'.format(', '.join(col))
        single = False
    else:
        single = True
    return list(
        arrconv(q[0]) if single else tuple(arrconv(a) for a in q) for q in
        db.query('SELECT {1} from error WHERE path in'
                 ' (SELECT path FROM opts WHERE {0})'.format(str(query), col)))


def opt_from_q(query, col):
    if not isinstance(col, str):
        col = '{}'.format(', '.join(col))
        single = False
    else:
        single = True

    return list(q[0] if single else q for q in
                db.query(
                    'SELECT {1} FROM opts WHERE ({0})'.format(str(query), col)))


from itertools import chain


def err_and_opts(query, col_opt, col_train, zipped=False, dikt=False):
    opts = opt_from_q(query, col_opt)
    errs = err_from_q(query, col_train)

    if not isinstance(errs[0], tuple):
        errs = (errs,)
    else:
        errs = tuple(zip(*errs))

    if not isinstance(opts[0], tuple):
        opts = (opts,)
    else:
        opts = tuple(zip(*opts))

    if dikt:
        headers = list(col_opt) if not isinstance(col_opt, str) else [col_opt]
        headers += list(col_train) if not isinstance(col_train, str) else [
            col_train]
        if zipped:
            return [dict(zip(headers, values))
                    for values in zip(*chain(opts, errs))]
        return dict(zip(headers, chain(opts, errs)))

    if zipped:
        l_opts = len(opts)
        return list((values[:l_opts], values[l_opts:])
                    for values in zip(*chain(opts, errs)))
    return opts, errs


def remove_bak():
    loaded = set(c[0].strip() for c in db.query('SELECT path FROM error'))
    rem_paths = [path for path in loaded if '.bak' in path]
    cond = Where().path.__contains__(rem_paths)
    db.execute('DELETE FROM error WHERE ' + str(cond))


def dump_root(root='/media/tethys/Data'):
    for folder in os.scandir(root):
        if folder.is_dir() and '.bak' not in folder.path:
            dump_folder_to_db(folder.path)


def gen_epoch_dat():
    try:
        add_col('error', 'bytea', 'train_mean')
        add_col('error', 'bytea', 'test_mean')
        add_col('error', 'bytea', 'train_std')
        add_col('error', 'bytea', 'test_std')
    except postgresql.exceptions.DuplicateColumnError:
        pass

    def proc_err(err, part):
        bpe = 255 if part == 'train' else 31
        tmp = np.array(err).reshape((-1, bpe))
        mean = tmp.mean(axis=1)
        std = tmp.std(axis=1)
        return mean, std, tmp

    insert_tr = insert_cmd('error', ('path', 'train_mean', 'train_std'))
    insert_val = insert_cmd('error', ('path', 'test_mean', 'test_std'))
    insert_optimum = insert_cmd('optimum',
                                ('path', 'epoch', 'train_err', 'test_err'))

    for (path,), (train, val) in err_and_opts('True', 'path', ('train', 'test'),
                                              zipped=True):
        print(path, train.shape)
        mean_tr, std, tr = proc_err(train, 'train')
        insert_tr(path, arrconv(mean_tr), arrconv(std))
        mean_te, std, te = proc_err(val, 'val')
        insert_val(path, arrconv(mean_te), arrconv(std))

        epoch = np.nanargmin(mean_te)
        insert_optimum(path, epoch, mean_tr[epoch], mean_te[epoch])


def gen_rnn_class():
    try:
        add_col('opts', 'text', 'arch')
    except postgresql.exceptions.DuplicateColumnError:
        pass
    insert = insert_cmd('opts', ('path', 'arch'))
    for path in opt_from_q('True', 'path'):
        if 'karpathlstm' in path:
            insert(path, 'LSTM')
        elif 'karpathgru':
            insert(path, 'GRU')
        else:
            raise ValueError(path)





def distinct(opt):
    return [q[0] for q in db.query('SELECT DISTINCT({0}) FROM opts '
                                   'ORDER BY {0} ASC'.format(opt))]





def lt_str(learning_rate):
    base = np.log10(learning_rate / 2)
    return '2e{0:1.1f}'.format(base)


def rect_lr():
    if 'lr' not in opt_cols:
        add_col('opts', 'real', 'lr')

    alphas = distinct('start_alpha')
    for alpha in alphas:
        base = np.log10(float(alpha) / 2)
        lr = (10 ** np.round(base, 2)) * 2
        db.prepare('UPDATE opts SET lr = $1 WHERE start_alpha = $2')(lr, alpha)

if __name__ == '__main__':
    import socket
    if socket.gethostname() == 'tethys':
        root = os.environ['HOME']
    else:
        root = '/media/tethys'
    dump_root(root + '/Data/')
    gen_epoch_dat()
    # scatter_optimum()
