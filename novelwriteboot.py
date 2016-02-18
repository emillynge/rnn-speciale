import numpy as np
import curses
from curses import wrapper
from contextlib import redirect_stdout
import io


class NovelWriter:
    def __init__(self, fr):
        self.fr = fr
        self.sentinels = list(fr.oven.charmap.str2arraygen(' \n,.:;'))
        self._f_pred = None
        self.outredir = io.StringIO()

    @property
    def f_pred(self):
        if self._f_pred is None:
            with redirect_stdout(self.outredir):
                self.fr.recipe.l_out
                self.fr.recipe.all_params
                self.fr.recipe.f_predict_single
                _f = self.fr.recipe.auto_predict
                self.fr.recipe.reset_hidden_states(batched=False)
            self._f_pred = _f

        return self._f_pred

    def main(self, stdscr):
        # Clear screen

        stdscr.clear()
        stdscr.addstr(0, 0, "Welcome to NovelWriter!", curses.A_STANDOUT)
        stdscr.addstr(0, 30, "prediction determinism: {0}".format(self.fr.opt.pred_sigma))
        stdscr.refresh()
        self.f_pred
        stdscr.addstr(1, 0, "just start typing :D (CTRL+E to exit)", curses.A_STANDOUT)
        stdscr.addstr(2, 0, "")
        key = stdscr.getkey()
        row = 3
        col = 0
        X = list()
        last_state = self.fr.recipe.get_hidden_state()
        while (key != curses.KEY_EXIT and key in self.fr.oven.charmap) or \
                        key in ['\t', '\x7f', 'KEY_UP', 'KEY_DOWN', 'KEY_RIGHT']:
            if key == '\t':
                stdscr.addstr(row, col, suggestion[:-1])
                col += len(suggestion) - 1
                key = suggestion[-1]
                X = list()
                last_state = self.fr.recipe.get_hidden_state()
                continue
            elif key == 'KEY_RIGHT':
                self.fr.recipe.set_hidden_state(last_state)
                if not X:
                    key = stdscr.getkey()
                    continue

            elif key in ('p', 'KEY_UP', 'KEY_DOWN'):
                rat = 2 if key == 'KEY_UP' else .5
                self.fr.opt.pred_sigma *= rat
                stdscr.addstr(0, 30, "prediction determinism: {0}".format(self.fr.opt.pred_sigma))
                self.fr.recipe.set_hidden_state(last_state)
                if not X:
                    key = stdscr.getkey()
                    continue

            elif ord(key.encode()) == 127: # BACKSPACE
                self.fr.recipe.set_hidden_state(last_state)
                if len(X) > 1:
                    X.pop()
                    col -= 1
                    stdscr.addstr(row, col, ' ')
            else:
                self.fr.recipe.set_hidden_state(last_state)
                x = self.fr.oven.charmap(key)
                X.append(x)
                stdscr.addstr(row, col, key)
                col += 1
            if key == '\n':
                row += 1
                col = 0

            out = np.array(list(self.f_pred(np.array(X), 100, sentinels=self.sentinels)),
                           dtype=bool)

            suggestion = self.fr.oven.charmap(out)
            stdscr.addstr(row+1, col, suggestion, curses.A_STANDOUT)
            key = stdscr.getkey(row, col)
            stdscr.addstr(row+1, col, ' ' * len(suggestion))


def main(fr):
    nw = NovelWriter(fr)
    try:
        wrapper(nw.main)
    except Exception:
        print(nw.outredir.getvalue())
        raise


if __name__ == '__main__':
    import sys
    from lasagnecaterer.menu import empty_fridge
    fr = empty_fridge(sys.argv[1])
    main(fr)
