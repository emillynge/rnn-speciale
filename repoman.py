from __future__ import unicode_literals, print_function
from git import Repo
from pprint import pformat
import logging
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
ppdeb(main_repo.submodules)



