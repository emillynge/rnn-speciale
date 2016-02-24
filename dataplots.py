from collections import OrderedDict

import sqldata as sd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import numpy as np

db = sd.db


def get_col_list(items, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    if isinstance(items, (int, float)):
        items = list(range(items))
    N = len(items)

    return OrderedDict((it, cmap.colors[int(i)]) for i, it in
                       zip(np.linspace(0, cmap.N - 1, N), items))


def scatter_optimum():
    alphas = sd.distinct('lr')
    baselines = sd.distinct('baseline')
    layers = sd.distinct('n_hid_lay')

    lay_cols = sd.get_col_list(layers)

    get = db.prepare('SELECT dropout, test_err FROM opts INNER JOIN '
                     'optimum ON optimum.path = opts.path AND '
                     '( opts.arch = $1 AND opts.baseline = $2 '
                     'AND opts.n_hid_lay = $3 AND opts.lr = $4)')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legend = OrderedDict()
    for alpha, col in lay_cols.items():
        # label = 'LR:' + lt_str(alpha)
        legend[alpha] = mlines.Line2D([], [], color=col, marker=None,
                                      linewidth=15, label=str(alpha))

    for symb, arch in [('o', 'LSTM'), ('*', 'GRU')]:
        mmarkers.MarkerStyle(symb)
        legend[arch] = mlines.Line2D([], [], color='k', marker=symb,
                                     linestyle=None,
                                     markersize=15, label=arch)

        for alpha, col in get_col_list(alphas).items():
            # col = [int(c * 255) for c in col]
            for baseline in baselines:

                for hid_lay in layers:
                    x, y = list(), list()
                    res = get(arch, baseline, hid_lay, alpha)
                    #                res = [it[:-2] for it in tmp
                    #                       if it[-1] == alpha]
                    if res:
                        do, te = tuple(zip(*res))
                        i = np.array(te).argmin()
                        x.append(alpha)  # do[i])
                        y.append(te[i])

                        ax.scatter(x, y, marker=symb, alpha=.5,
                                   facecolors=[lay_cols[hid_lay]] * len(x),
                                   s=(np.log2(float(baseline)) - np.log2(
                                       64)) * 200)

    ax.set_ylim([1.1, 1.5])
    ax.legend(handles=list(legend.values()))
    ax.get_xaxis().set_ticks(alphas)
    ax.get_xaxis().set_ticklabels([lt_str(a) for a in alphas])
    plt.show()
