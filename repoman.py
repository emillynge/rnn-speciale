from __future__ import unicode_literals, print_function
from git import Repo, Submodule, UpdateProgress, Diff, Tree, DiffIndex
from pprint import pformat
import logging
import re
from functools import wraps
logging.basicConfig(level=logging.DEBUG)
deb = logging.debug

@wraps(deb)
def ppdeb(*objects, **kwargs):
    if len(objects) == 1:
        objects = objects[0]
    label = kwargs.pop('label', None)
    msg = label + ':' if label else ''
    msg += '\n' + pformat(objects, indent=4,)
    deb(msg, **kwargs)

main_repo = Repo('.')

class MyProgress(UpdateProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            logging.info(message)
progress = MyProgress()

def first_commit(sr):
    for commit in sr.iter_commits():
        pass
    return commit

def do_fetch():
    for sub_mod in main_repo.submodules:
        if sub_mod.module_exists():
            sr = Repo(sub_mod.abspath)
            sr.git.checkout(sub_mod.branch.name)
            if 'localcompiled' not in sr.branches:
                sr.create_head('localcompiled', 'HEAD')
            sub_mod.update(recursive=True, to_latest_revision=True, progress=progress, dry_run=False)
        else:
            sub_mod.update(recursive=True, to_latest_revision=True, progress=progress, dry_run=False)
            sr = Repo(sub_mod.abspath)
            sr.create_head('localcompiled', first_commit)



def do_compile():
    for sub_mod in main_repo.submodules:
        sr = Repo(sub_mod.abspath)
        sr.git.checkout(sub_mod.branch.name)
        if 'localcompiled' not in sr.branches:
            raise ValueError('no localcompiled branch. Cannot determine diff')
        comp_branch = [branch for branch in sr.branches if branch.name == 'localcompiled']
        if not comp_branch:
            raise ValueError('branch does not exist')
        comp_branch = comp_branch[0]
        sr.commit().diff(comp_branch.commit)


do_fetch()

#
#
#
# c_detect = re.compile('\.[ch]p{0,2}$')
# def adiffer(old_commit):
#     for dif in sr.commit().diff(old_commit):
#         assert isinstance(dif, Diff)
#         if c_detect.search(dif.a_path) or c_detect.search(dif.b_path):
#             deb(dif.a_path + ' | ' + dif.b_path)
#         deb(dif.a_path)
#
# adiffer(100)
# assert isinstance(sub_repo, Submodule)
# ppdeb(sub_repo)
#
#
# hcomm = main_repo.head.commit
# hcomm.diff('HEAD~1')
# #sub_repo.
