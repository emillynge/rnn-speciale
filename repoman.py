from __future__ import unicode_literals, print_function
from git import Repo, Submodule, UpdateProgress, Diff, Tree, DiffIndex
from pprint import pformat
import logging
import re
from collections import defaultdict, deque
from functools import wraps

logging.basicConfig(level=logging.DEBUG)
deb = logging.debug
from argparse import ArgumentParser
import socket
import sys

if sys.version_info.major == 3:
    from io import StringIO
    pip = 'pip3.5'
else:
    from StringIO import StringIO
    pip = 'pip2'


HOST = socket.gethostname()
import os
from subprocess import Popen, PIPE

_PREBUILD = {'pycuda': {('archlaptop',): NotImplemented,
                        ('tethys',): [],
                        ('n-62-12-1',): [['python', 'configure.py', '--cudadrv-lib-dir=' + os.getcwd() + '/lib',
                                          '--boost-inc-dir=/appl/boost/1.57.0/lib',
                                          '--boost-inc-dir=/appl/boost/1.57.0/include',
                                          '--boost-compiler=gcc47',
                                          '--boost-python-libname=boost_python']]}}
PREBUILD = defaultdict(dict)
for submod, host_confs in _PREBUILD.items():
    for hosts, conf in host_confs.items():
        for host in hosts:
            PREBUILD[submod][host] = conf

SITEPACKAGES = os.environ.get('HOME') + '/.local/lib/python2.7/site-packages'


class DummyProc(object):
    def poll(self):
        return 0

    @property
    def stdout(self):
        return StringIO('Dummymethod')

    @property
    def stderr(self):
        return StringIO('Dummymethod')


def get_prebuild(sub_mod):
    if sub_mod not in PREBUILD:
        return list()

    confs = PREBUILD[sub_mod]
    if HOST not in confs:
        return list()

    build = confs[HOST]
    if build is NotImplemented:
        raise NotImplementedError
    return build


@wraps(deb)
def ppdeb(*objects, **kwargs):
    if len(objects) == 1:
        objects = objects[0]
    label = kwargs.pop('label', None)
    msg = label + ':' if label else ''
    msg += '\n' + pformat(objects, indent=4, )
    deb(msg, **kwargs)


main_repo = Repo('.')


class MyProgress(UpdateProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            logging.info(message)


progress = MyProgress()


# noinspection PyUnboundLocalVariable
def first_commit(sr):
    for commit in sr.iter_commits():
        pass
    return commit


c_detect = re.compile('\.[ch]p{0,2}$')


def compile_changes_ocurred(master, localcompiled):
    for diff in master.commit().diff(localcompiled.commit):
        if c_detect.search(diff.a_path) or c_detect.search(diff.b_path):
            ppdeb(diff.a_path, label='Compile change')
            return True
    return False


def do_update(sub_modules=None):
    sub_mods = [_sm for _sm in main_repo.submodules if sub_modules is None or _sm.name in sub_modules]
    for sub_mod in sub_mods:
        if sub_mod.module_exists():
            sr = Repo(sub_mod.abspath)
            sr.git.checkout(sub_mod.branch.name)
            if 'localcompiled' not in sr.branches:
                sr.create_head('localcompiled', 'HEAD')
            sub_mod.update(recursive=True, to_latest_revision=True, progress=progress, dry_run=False, force=True)
        else:
            sub_mod.update(recursive=True, to_latest_revision=True, progress=progress, dry_run=False, force=True)
            sr = Repo(sub_mod.abspath)
            sr.create_head('localcompiled', first_commit(sr))


COMPILEQUEUE = deque()


def do_compile(sub_modules=None, force=False):
    sub_mods = [sm for sm in main_repo.submodules if sub_modules is None or sm.name in sub_modules]

    for sub_mod in sub_mods:
        sr = Repo(sub_mod.abspath)
        sr.git.checkout(sub_mod.branch.name)
        if 'localcompiled' not in sr.branches:
            raise ValueError('no localcompiled branch. Cannot determine diff')
        comp_branch = [branch for branch in sr.branches if branch.name == 'localcompiled']
        if not comp_branch:
            raise ValueError('branch does not exist')
        comp_branch = comp_branch[0]
        if compile_changes_ocurred(sr, comp_branch) or force:
            try:
                prebuild = get_prebuild(sub_mod.name)
            except NotImplementedError:
                logging.error('submodule {0} cannot be compiled on host {1}'.format(sub_mod.name, HOST))
                continue

            COMPILEQUEUE.append((DummyProc(), prebuild, sub_mod.abspath))

    while COMPILEQUEUE:
        proc, prebuild, subpath = COMPILEQUEUE.popleft()
        if proc.poll() is not None:
            if proc.poll() != 0:
                raise RuntimeError(pformat({'StdOut': proc.stdout.read(),
                                            'StdErr': proc.stderr.read()}))
            if prebuild:                    # still commands left in prebuild
                commands = prebuild.pop(0)
                proc = Popen(commands, stdout=PIPE, stderr=PIPE, cwd=subpath)
                ppdeb(subpath, commands, label='Prebuild command')
            elif prebuild is not None:      # no commands left. run easyinstall
                proc = Popen([pip, 'install', subpath, '--user'], stdout=PIPE, stderr=PIPE)
                prebuild = None
                ppdeb(subpath, label="Pip'ing")
            else:                           # easyinstall has completed. skip putting into queue and mark as compiled
                sr = Repo(subpath)
                for head in sr.heads:
                    if head.name == 'localcompiled':
                        head.reference = sr.head.reference
                logging.info('{0} recompiled and installed'.format(subpath))
                continue
        COMPILEQUEUE.append((proc, prebuild, subpath))


parser = ArgumentParser(description="update git submodules and recompile if necesarry")
parser.add_argument('--sub-modules',
                    '-s',
                    nargs='*',
                    help='specify which submodules to pull (default: all)',
                    dest='submodules')

parser.add_argument('--force-rebuild',
                    '-bf',
                    action="store_true",
                    help='rebuild after pull',
                    dest="force_rebuild")

if __name__ == "__main__":
    args = parser.parse_args()
    do_update(args.submodules)
    do_compile(args.submodules, args.force_rebuild)
