from subprocess import call, Popen, PIPE
from argparse import ArgumentParser
from QsubTools import WORKDIR, SITE_PACKAGE_DIR
import os
import re

BUILD_COMMANDS = {'Theano': [['python', 'setup.py', 'build'],
                             ['rm', '-rf', SITE_PACKAGE_DIR + '/theano'],
                             ['cp', '-rf', 'build/lib/theano', SITE_PACKAGE_DIR + '/theano']],
                  'nntools': [['python', 'setup.py', 'build'],
                              ['rm', '-rf', SITE_PACKAGE_DIR + '/nntools'],
                              ['cp', '-rf', 'build/lib/lasagne', SITE_PACKAGE_DIR + '/nntools']],
                  'skaae': [['python', 'setup.py', 'build'],
                            ['rm', '-rf', SITE_PACKAGE_DIR + '/skaae'],
                            ['cp', '-rf', 'build/lib/lasagne', SITE_PACKAGE_DIR + '/lasagne']],
                  'pycuda': [['git', 'submodule', 'update', '--init'],
                             ['rm ', '-f', 'siteconf.py'],
                             ['python', 'configure.py', '--cudadrv-lib-dir=' + WORKDIR + '/lib'],
                             ['python', 'setup.py', 'build'],
                             ['rm', '-rf', SITE_PACKAGE_DIR + '/pycuda'],
                             ['cp', '-rf', 'build/lib.linux-x86_64-2.7/pycuda', SITE_PACKAGE_DIR + '/pycuda']],
                  }

parser = ArgumentParser(description="update git submodules and optionally recompile")
parser.add_argument('--sub-modules',
                    '-s',
                    nargs='*',
                    help='specify which submodules to pull (default: all)',
                    dest='submodules')

parser.add_argument('--rebuild',
                    '-b',
                    action="store_true",
                    help='rebuild if necessary after pull',
                    dest="rebuild")

parser.add_argument('--force-rebuild',
                    '-bf',
                    action="store_true",
                    help='rebuild after pull',
                    dest="force_rebuild")

parser.add_argument('--fetch-libs',
                    '-l',
                    action="store_true",
                    help='fetch needed libraries not present on standard installation',
                    dest="fetch_libs")


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
                                      'branch': 'master',
                                      'name': header}
            else:
                field, val = regexp_field_val.findall(line)[0]
                submodules[header][field] = val
    return submodules


def check_submodules(k_sm, p_sm):
    submodule_paths = dict((sm['path'], key) for (key, sm) in k_sm.items())
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


def build_module(submodule):
    os.chdir(submodule['path'])
    try:
        for cmd in BUILD_COMMANDS[submodule['name']]:
            p_call(cmd)
    finally:
        os.chdir('../')


def p_call(command):
    print('calling: {0}'.format(' '.join(command)))
    p = Popen(command, stdout=PIPE, stderr=PIPE)
    while p.poll() is None:
        out_line = p.stdout.readline()
        if out_line:
            print out_line[:-1]
        error_line = p.stderr.readline()
        if error_line:
            print '# Error - %s' % error_line[:-1]


def pull_submodule(submodule_dict):
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
        submodule_dict['rebuild'] = True
    finally:
        os.chdir('../')


def get_libs():
    if not os.path.exists('lib'):
        os.mkdir('lib')
    from StringIO import StringIO
    from rpmfile import RPMFile
    from urllib import urlopen

    repo = "http://developer.download.nvidia.com/compute/cuda/repos/rhel6/"
    pkg_list = urlopen(repo + "x86_64").read()
    pkg_url = re.findall('x86_64/cuda-driver-dev-6-5.+?\.rpm', pkg_list)[-1]  # last element gets the latest version
    rpm_buffer = StringIO(urlopen(repo + pkg_url).read())
    rpm_file = RPMFile(name=pkg_url.split('/')[-1], fileobj=rpm_buffer)
    members = rpm_file.getmembers()
    libcuda = None
    for member in members:
        if "libcuda.so" in member.name:
            libcuda = rpm_file.extractfile(member)
            break
    if not libcuda:
        raise Exception('libcuda.so not found in rpm file "{0}"'.format(rpm_file.headers['name']))
    with open('lib/libcuda.so', 'w') as fp:
        fp.write(libcuda.read())
    c_dir = os.path.abspath('.')
    p_call(['ln', '-s', c_dir + '/lib/libcuda.so', c_dir + '/lib/libcuda.so.1'])

    print('fetched ')


if __name__ == "__main__":
    args = parser.parse_args()

    known_submodules = read_gitmodules()
    if args.submodules:
        pull_submodules = [sm.strip('/') for sm in args.submodules]
        check_submodules(known_submodules, pull_submodules)
    else:
        pull_submodules = known_submodules.keys()

    if args.fetch_libs:
        get_libs()

    for sm in pull_submodules:
        sm_info = known_submodules[sm]
        pull_submodule(sm_info)
        if (sm_info.get('rebuild', False) and args.rebuild) or args.force_rebuild:
            build_module(sm_info)





