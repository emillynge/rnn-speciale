from subprocess import call, Popen, PIPE
from argparse import ArgumentParser
from collections import namedtuple
import os
import re

parser = ArgumentParser(description="update git submodules and optionally recompile")
parser.add_argument('--sub-modules',
                    '-s',
                    nargs='*',
                    help='specify which submodules to pull (default: all)',
                    dest='submodules')

parser.add_argument('--rebuild',
                    '-b',
                    action="store_true",
                    help='rebuild if necessary after pull')

parser.add_argument('--force-rebuild',
                    '-bf',
                    action="store_true",
                    help='rebuild after pull')

def read_gitmodules():
    regexp_header = re.compile('\[submodule "(\w+)"\]')
    regexp_field_val = re.compile('[ \t]*(.+) = (.+)')
    submodules = dict()
    with open('.gitmodules', 'r') as fp:
        for line in fp:
            _header = regexp_header.findall(line)
            if _header:
                header = _header[0]
                submodules[header] = {'url': None,
                                      'path': None,
                                      'branch': 'master'}
            else:
                field, val = regexp_field_val.findall(line)[0]
                submodules[header][field] = val
    return submodules


def check_submodules(k_sm, p_sm):
    submodule_paths = dict((sm['path'], key) for(key, sm) in k_sm.items())
    changes = list()
    for submodule in p_sm:
        print "HIII %r" % submodule
        if submodule not in k_sm:
            if submodule in submodule_paths:
                changes.append((submodule, submodule_paths[submodule]))
            else:
                raise ValueError('submodule %s scheduled for pull does not exist'.format(submodule))

    for (orig_name, new_name) in changes:
        p_sm.remove(orig_name)
        p_sm.append(new_name)


def p_call(command):
        p = Popen(command, stdout=PIPE, stderr=PIPE)
        while p.poll() is None:
            out_line = p.stdout.readline()
            if out_line:
                print out_line[:-1]
            error_line = p.stderr.readline()
            if error_line:
                print '# Error - %s' % error_line[:-1]

def pull_submodule(submodule_dict, rebuild='adhoc'):
    os.chdir(submodule_dict['path'])
    try:
        print "--CHECKOUT--"
        p_call(['git', 'checkout', submodule_dict['branch']])
        print "--COMMIT--"
        p_call(['git', 'commit', '-a', '-m', '"scrap changes"'])
        print "--MERGE--"
        p_call(['git', 'merge', '-X', 'theirs', submodule_dict['branch']])
        print "--PULL--"
        p_call(['git', 'pull', 'origin', submodule_dict['branch']])
    finally:
        os.chdir('../')


args = parser.parse_args()

known_submodules = read_gitmodules()
if args.submodules:
    pull_submodules = [sm.strip('/') for sm in args.submodules]
    check_submodules(known_submodules, pull_submodules)
else:
    pull_submodules = known_submodules.keys()

for sm in pull_submodules:
    pull_submodule(known_submodules[sm])



