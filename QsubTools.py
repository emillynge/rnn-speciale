import os
import datetime

__author__ = 'emil'
USER = "s082768@student.dtu.dk"
WORKDIR = "/zhome/25/2/51526/Speciale/rnn-speciale"
SITE_PACKAGE_DIR = "zhome/25/2/51526/.local/lib/python2.7/site-packages"
LOGDIR = "/zhome/25/2/51526/Speciale/rnn-speciale/qsub/logs"
SERVER_PYTHON_BIN = "/usr/local/gbar/apps/python/2.7.1/bin/python"
SERVER_MODULE_BIN = "/apps/dcc/bin/module"
QSUB_MANAGER_PORT = 5000
SSH_USERNAME = "s082768"
SSH_PRIVATE_KEY = "/home/emil/.ssh/id_rsa"
SSH_HOST = "hpc-fe1.gbar.dtu.dk"
SSH_PORT = 22

BASE_MODULES = {"python": '', "cuda": '', "boost": ''}
from subprocess import Popen, PIPE

FNULL = open('/dev/null', 'w')
from argparse import ArgumentParser
import re
import Pyro4.socketutil
from Pyro4 import errors as pyro_errors
import logging
import sys
import json
from copy import copy
from time import sleep

Pyro4.config.COMMTIMEOUT = 5.0  # without this daemon.close() hangs

from collections import namedtuple, defaultdict


class InvalidQsubArguments(Exception):
    pass


def open_ssh_session_to_server():
    import pxssh
    s = pxssh.pxssh()
    if not s.login(SSH_HOST, SSH_USERNAME, ssh_key=SSH_PRIVATE_KEY):
        raise pxssh.ExceptionPxssh('Login failed')
    return s


def create_logger(logger_name="Qsub", log_to_file='logs/qsubs.log', log_to_stream=True, log_level='DEBUG',
                  format_str=None):
    if not log_to_file and not log_to_stream:   # neither stream nor logfile specified. no logger wanted.
        return DummyLogger()
    # create logger with 'spam_application'
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(log_level)
    # create formatter and add it to the handlers
    if not format_str:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)

    if log_to_file:
        if not isinstance(log_to_file, list):
            log_to_file = [log_to_file] if log_to_file else list()

        for logfile in log_to_file:
            # create file handler which logs even debug messages
            fh = logging.FileHandler(logfile)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            _logger.addHandler(fh)

    if log_to_stream:
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        _logger.addHandler(ch)

    return _logger


logger = create_logger()


class DummyLogger(object):
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def make_tunnel(port, server_host="127.0.0.1"):
    from sshtunnel import SSHTunnelForwarder

    server = SSHTunnelForwarder(
        (SSH_HOST, SSH_PORT),
        ssh_username=SSH_USERNAME,
        ssh_private_key=SSH_PRIVATE_KEY,
        remote_bind_address=(server_host, port),
        logger=DummyLogger(),
        raise_exception_if_any_forwarder_have_a_problem=False)
    server.start()
    return server, server.local_bind_port


HPC_Time = namedtuple("Time", ['h', 'm', 's'])
HPC_Time.__new__.__defaults__ = (0, 0, 0)
HPC_resources = namedtuple("Resource", ['nodes', 'ppn', 'gpus', 'pvmem', 'vmem'])
HPC_resources.__new__.__defaults__ = (1, 1, 0, None, None)


class QsubClient(object):
    def __init__(self, restart_manager=False):
        self.max_retry = 3
        self.logger = create_logger("Client")

        self.manager_ip, hot_start = RemoteQsubCommandline('-i isup manager').get('ip', 'return')
        if not hot_start:
            self.start_manager()

        self.manager_ssh_server, self.manager_client_port = make_tunnel(5000, server_host=self.manager_ip)
        self.manager = self.get_manager()
        self.manager.is_alive()
        self.logger.debug("Successfully connected to Qsub manager on  {1}:{0}".format(self.manager_client_port,
                                                                                      self.manager_ip))

        if restart_manager and hot_start:
            self.restart_manager()

    def generator(self, package, module, wallclock, resources, rel_dir="", additional_modules=None):
        qsub_gen = QsubGenerator(self.manager, package, module, wallclock, resources, rel_dir, additional_modules)
        return qsub_gen.get_instance_class()

    def start_manager(self):
        self.manager_ip = RemoteQsubCommandline('-i start manager').get('ip')
        # timeout = retries * 2
        # RemoteQsubCommandline('init manager')
        # sleep(timeout)
        # if not self.isup_manager():
        #     self.logger.info("Manager still not up after init")
        #     if retries > self.max_retry:
        #         self.logger.error("Giving up on initializing manager after {0} tries".format(self.max_retry))
        #     else:
        #         self.start_manager(retries=retries + 1)

    def get_manager(self):
        return Pyro4.Proxy("PYRO:qsub.manager@localhost:{0}".format(self.manager_client_port, self.manager_ip))

    def restart_manager(self):
        self.manager.shutdown()
        try:
            while True:
                self.manager.is_alive()
        except pyro_errors.CommunicationError:
            pass
        self.start_manager()
        self.logger.info("Manager restarted")


class QsubManager(object):
    def __init__(self):
        self.logger = create_logger('Manager')
        self._available_modules = self.get_available_modules()
        self.running = True
        self.latest_sub_id = -1
        self.qsubs = defaultdict(dict)
        self.logger.info("Manager stated")

    def available_modules(self):
        return self._available_modules

    def request_submission(self):
        self.latest_sub_id += 1
        self.qsubs[self.latest_sub_id]['state'] = 'requested'
        self.logger.info("Submission id: {0} granted".format(self.latest_sub_id))
        return self.latest_sub_id, self.logfile(self.latest_sub_id)

    def stage_submission(self, sub_id, script):
        with open('qsubs/{0}.sh'.format(sub_id), 'w') as fp:
            fp.write(script)
        self.qsubs[sub_id]['state'] = 'staged'

    def get_state(self, sub_id):
        return self.qsubs[sub_id]['state']

    @staticmethod
    def logfile(sub_id):
        return '{0}/{1}'.format(LOGDIR, sub_id)

    def error_log(self, sub_id):
        return self.logfile(sub_id) + '.e'

    def out_log(self, sub_id):
        return self.logfile(sub_id) + '.o'

    @staticmethod
    def get_available_modules():
        p = Popen("module avail", stdout=FNULL, stderr=PIPE, shell=True)
        lines = re.findall('/apps/dcc/etc/Modules/modulefiles\W+(.+)',
                           p.communicate()[1], re.DOTALL)[0]
        logger.debug(lines)
        modules = re.split('[ \t\n]+', lines)[:-1]
        module_ver_list = [m.strip('(default)').split('/') for m in modules]

        module_dict = defaultdict(list)
        for mod_ver in module_ver_list:
            if len(mod_ver) < 2:
                mod_ver.append('default')

            module_dict[mod_ver[0]].append(mod_ver[1])
        return module_dict

    def is_alive(self):
        return self.running

    def shutdown(self):
        print 'shutting down qsub manager'
        self.running = False

    @staticmethod
    def path_exists(path):
        return os.path.exists(path)

    @staticmethod
    def get_ip():
        return Pyro4.socketutil.getIpAddress('localhost', workaround127=True)


class QsubGenerator(object):
    def __init__(self, qsub_manager, package, module, wallclock, resources, rel_dir, additional_modules):
        assert isinstance(wallclock, HPC_Time)
        assert isinstance(resources, HPC_resources)
        assert isinstance(qsub_manager, (QsubManager, Pyro4.Proxy))

        self.available_modules = qsub_manager.available_modules()
        self.resources = None
        self.wc_time = None
        self.modules = copy(BASE_MODULES)
        self.base_dir = WORKDIR
        self.rel_dir = rel_dir
        self.package = package
        self.module = module
        self.manager_ip = qsub_manager.get_ip()
        self.resources = resources
        self.wc_time = wallclock
        self.logger = create_logger('Generator')
        self.manager = qsub_manager

        try:
            if resources.nodes < 1 or resources.ppn < 1:
                raise InvalidQsubArguments('A job must have at least 1 node and 1 processor')

            if not any(wallclock):
                raise InvalidQsubArguments('No wall clock time assigned to job: {0}:{1}:{2}'.format(wallclock))

            if not self.manager.path_exists(self.work_dir):
                raise InvalidQsubArguments("Work directory {0} doesn't exist.".format(self.work_dir))

            if additional_modules:
                self.modules.update(additional_modules)
            self.check_modules()
        except InvalidQsubArguments as e:
            self.logger.error('Invalid parameters passed to Qsub', exc_info=True)
            raise e

        self.submission_script = self.make_submission_script()

    def make_submission_script(self):
        ss = SubmissionScript(self.work_dir, self.modules)
        ss.resources(self.resources)
        ss.wallclock(self.wc_time)
        ss.name('{0}.{1}'.format(self.package, self.module))
        ss.mail(SSH_USERNAME + '@student.dtu.dk')
        return ss

    @property
    def work_dir(self):
        return self.base_dir + '/' + self.rel_dir

    def check_modules(self):
        for (module, version) in self.modules.iteritems():
            if module not in self.available_modules:
                raise InvalidQsubArguments("Required module {0} is not available".format(module))
            if version and version not in self.available_modules[module]:
                raise InvalidQsubArguments("Required module version {0} is not available for module {1}".format(version,
                                                                                                                module))
            self.logger.debug("module {0}, version {1} is available".format(module, version if version else "default"))

    def get_instance_class(self):
        manager = self.manager
        ss = self.submission_script

        class QsubInstance(BaseQsubInstance):
            @staticmethod
            def set_manager():
                return manager

            @staticmethod
            def set_submission_script():
                return ss

            @staticmethod
            def set_qsub_generator():
                return self

        return QsubInstance


class SubmissionScript(object):
    def __init__(self, work_dir, modules):
        self.lines = ["#!/bin/sh"]
        self.wd = work_dir
        self.modules = modules

    def generate(self, execute_commands, log_file):
        script = '\n'.join(self.lines) + '\n'
        script += self.make_pbs_pragma('e', log_file + ".e") + '\n'
        script += self.make_pbs_pragma('o', log_file + ".o") + '\n'

        for module_name, version in self.modules.iteritems():
            script += 'module load ' + module_name
            if version:
                script += '/' + version
            script += '\n'

        script += 'cd {0}\n'.format(self.wd)

        if isinstance(execute_commands, list):
            script += '\n'.join(execute_commands)
        else:
            script += execute_commands
        return script

    @staticmethod
    def make_pbs_pragma(flag, line):
        return "#PBS -" + flag.strip(' ') + ' ' + line

    def append_pbs_pragma(self, flag, line):
        self.lines.append(self.make_pbs_pragma(flag, line))

    def name(self, name):
        self.append_pbs_pragma('N ', name)

    def mail(self, mail_address):
        self.append_pbs_pragma('m', mail_address)

    def resources(self, resources):
        assert isinstance(resources, HPC_resources)
        self.append_pbs_pragma('l', 'nodes={1}:ppn={0}'.format(resources.ppn, resources.nodes))

        if resources.gpus:
            self.append_pbs_pragma('l', 'gpus={0}'.format(resources.gpus))

        if resources.pvmem:
            self.append_pbs_pragma('l', 'pvmem={0}'.format(resources.pvmem))

        if resources.vmem:
            self.append_pbs_pragma('l', 'vmem={0}'.format(resources.vmem))

    def wallclock(self, wallclock):
        assert isinstance(wallclock, HPC_Time)
        self.append_pbs_pragma("l", "walltime={0}:{1}:{2}".format(wallclock.h, wallclock.m, wallclock.s))


class BaseQsubInstance(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.qsub_manager = self.set_manager()
        self.qsub_client = self.set_qsub_client()
        self.submission_script = self.set_submission_script()
        self.qsub_generator = self.set_qsub_generator()
        (self.sub_id, self.logfile) = self.qsub_manager.request_submission()

    def stage_submission(self):
        exe = """python -c "from {0} import {1}; from QsubTools import QsubExecutor;
        QsubExecutor({1}, {2}, '{3}')" """.format(
            self.qsub_generator.package, self.qsub_generator.module, self.sub_id, self.qsub_manager.get_ip())
        script = self.submission_script.generate(exe, self.logfile)
        self.qsub_manager.stage_submission(self.sub_id, script)

    @staticmethod
    def set_qsub_client():
        return QsubClient()

    @staticmethod
    def set_manager():
        return QsubManager()

    @staticmethod
    def set_submission_script():
        return SubmissionScript(WORKDIR, BASE_MODULES)

    @staticmethod
    def set_qsub_generator():
        return QsubGenerator(qsub_manager=None, package=None, module=None, wallclock=HPC_Time(),
                             resources=HPC_resources(), rel_dir="", additional_modules={})


class QsubExecutor(object):
    def __init__(self, cls, sub_id, manager_ip):
        self.manager = Pyro4.Proxy("PYRO:qsub.manager@{0}:5000".format(manager_ip))
        if self.manager.is_alive():
            print "manager found!"
        print cls
        print sub_id


class QsubCommandline(object):
    def __init__(self):
        self.args2method = {'manager': {'start': self.start_manager,
                                        'stop': self.stop_manager,
                                        'isup': self.isup_manager}}
        self.stdout_logger = create_logger('CLI/stdout', log_to_file="", log_to_stream=True, format_str='%(message)s')
        self.args = self.parse_args()
        if self.args.remote:
            self.data = RemoteQsubCommandline(' '.join([arg for arg in sys.argv[1:] if arg not in ['-r', '--remote']]))
        else:
            self.logger = self.create_logger()

            self.data = dict()

            self.pre_execute()
            self.execute()
            self.post_execute()

    def get(self, *args):
        return tuple(self.data[key] for key in args)

    def parse_args(self, *args):
        return self.get_argument_parser().parse_args()

    def get_exec_func(self):
        return self.args2method[self.args.module][self.args.action]

    def pre_execute(self):
        self.data['ip'] = self.get_ip()
        if self.args.ip:
            self.stdout('ip', self.data['ip'])

    def execute(self):
        try:
            self.get_exec_func().__call__()
        except Exception as e:
            self.logger.error("Exception occurred during execution of {0} {1}".format(self.args.action,
                                                                                      self.args.module),
                              exc_info=True)
            sleep(.5)
            self.stdout('error', e.message)

    def post_execute(self):
        pass

    @staticmethod
    def get_ip():
        return QsubManager.get_ip()

    def get_logger_args(self):
        kwargs = dict()
        kwargs["log_to_stream"] = self.args.stream
        if self.args.logfiles is False:
            kwargs['log_to_file'] = list()
        else:
            if self.args.logfiles:
                kwargs['log_to_file'] = self.args.logfiles
        kwargs["logger_name"] = 'CLI/{0} {1}'.format(self.args.action, self.args.module)
        if self.args.log_level:
            kwargs['log_level'] = self.args.log_level
        return kwargs

    def create_logger(self):
        kwargs = self.get_logger_args()
        return create_logger(**kwargs)

    def stdout(self, tag, obj):
        self.stdout_logger.debug("{0}: {1}\n\r".format(tag, json.dumps(obj)))

    def get_manager(self):
        return Pyro4.Proxy("PYRO:qsub.manager@{0}:{1}".format(self.data['ip'], QSUB_MANAGER_PORT))

    def execute_return(self, result):
        self.stdout('return', result)

    def start_manager(self):
        self.logger.debug("Initializing manager")
        daemon = Pyro4.Daemon(port=QSUB_MANAGER_PORT, host=self.data['ip'])
        self.logger.debug("Init Manager")
        manager = QsubManager()
        daemon.register(manager, "qsub.manager")
        self.logger.info("putting manager in request loop")
        self.stdout('blocking', datetime.datetime.now().isoformat())
        daemon.requestLoop(loopCondition=manager.is_alive)

    def stop_manager(self):
        try:
            manager = self.get_manager()
            manager.shutdown()
            while True:
                manager.is_alive()
        except pyro_errors.CommunicationError:
            self.execute_return(0)

    def isup_manager(self):
        manager = self.get_manager()
        try:
            if manager.is_alive():
                self.execute_return(True)
        except pyro_errors.CommunicationError:
            self.execute_return(False)

    @staticmethod
    def get_argument_parser():
        parser = ArgumentParser('Command line interface to QsubTools')
        parser.add_argument('-i', '--get-ip',
                            action='store_true',
                            help='output ip to stdout before executing action',
                            dest='ip')

        parser.add_argument('-r', '--remote',
                            action='store_true',
                            help='execute this command on remote server',
                            dest='remote')

        logging_group = parser.add_argument_group("logging")
        logging_group.add_argument('-s', '--stream', action='store_true',
                                   help='activate logging to stdout (default False)',
                                   dest='stream')

        logging_group.add_argument('-L', '--log-level',
                                   choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
                                   help='log level',
                                   dest='log_level')

        # Logfile specification
        logfile_group = logging_group.add_mutually_exclusive_group()
        logfile_group.add_argument('-f', '--file',
                                   action='append',
                                   help='specify logging to specific file(s).',
                                   dest='logfiles', nargs='+')

        logfile_group.add_argument('-F', '--disable-file',
                                   action='store_false',
                                   help='disable logging to file',
                                   dest='logfiles')

        # Required arguments
        parser.add_argument('action',
                            choices=['start', 'stop', 'isup'],
                            help="action to send to module")

        parser.add_argument('module',
                            choices=['manager'],
                            help="module to send action to")
        return parser

    @staticmethod
    class CommandLineException(Exception):
        pass


class RemoteQsubCommandline(QsubCommandline):
    def __init__(self, commands):
        self.command = commands
        self.ssh = self.setup_ssh_instance()
        self.ssh.ignore_sighup = False
        super(RemoteQsubCommandline, self).__init__()

    def parse_args(self):
        return self.get_argument_parser().parse_args(self.command.split())

    def create_logger(self):
        return DummyLogger()

    def blocking(self):
        if self.args.action == 'start':
            return True
        return False

    def pre_execute(self):
        if self.args.stream:
            self.ssh.logfile = sys.stdout

        blocking = self.blocking()
        full_command = ""
        if blocking:
            full_command += 'nohup '

        full_command += "python QsubTools.py "
        full_command += self.command

        if blocking:
            full_command += ' > nohup.out & tail -f nohup.out'
        self.command = full_command

    def execute(self):
        self.ssh.sendline(self.command)

    def ssh_expect(self, pattern):
        patterns = ['error:', pattern]
        idx = self.ssh.expect(patterns)
        line = self.ssh.readline().strip('[\n\r :]')
        if idx == 0:
            self.data['error'] = line
            sleep(.5)
            raise self.CommandLineException('in {0}\n\t{1}'.format(self.get_exec_func().im_func.func_name,
                                                                   self.data['error']))
        else:
            self.data[patterns[idx].strip(':')] = json.loads(line)

    def post_execute(self):
        if self.args.ip:
            self.ssh_expect('ip:')

        if not self.blocking():
            self.ssh_expect('return:')
            self.ssh.logout()
            self.ssh.terminate()
        else:
            self.ssh_expect('blocking')
            self.ssh.terminate()

    @staticmethod
    def setup_ssh_instance():
        s = open_ssh_session_to_server()
        s.sendline('cd {0}'.format(WORKDIR))
        s.prompt()
        return s

if __name__ == "__main__":
    QsubCommandline()
