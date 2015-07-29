import os

__author__ = 'emil'
USER = "s082768@student.dtu.dk"
WORKDIR = "/zhome/25/2/51526/Speciale/rnn-speciale"
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
import re
import Pyro4.socketutil
from Pyro4 import errors as pyro_errors
import logging
import sys
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

def create_logger(loggername="Qsub"):
    # create logger with 'spam_application'
    _logger = logging.getLogger(loggername)
    _logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('logs/qsubs.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    _logger.addHandler(fh)
    _logger.addHandler(ch)
    return _logger


logger = create_logger()


class DummyLogger(object):
    def debug(self, *args):
        pass

    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def error(self, *args):
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


def init_manager():
    _logger = create_logger('init')
    _logger.debug("Initializing manager")
    try:
        daemon = Pyro4.Daemon(port=QSUB_MANAGER_PORT)
        _logger.debug("Init Manager")
        manager = QsubManager()
        daemon.register(manager, "qsub.manager")
        _logger.info("putting manager in request loop")
        daemon.requestLoop(loopCondition=manager.is_alive)
    except Exception as e:
        _logger.error(e.message, exc_info=True)
        raise e


def isup_manager():
    manager = Pyro4.Proxy("PYRO:qsub.manager@localhost:5000")
    try:
        if manager.is_alive():
            print True
    except pyro_errors.CommunicationError:
        print False


class QsubClient(object):
    def __init__(self, restart_manager=False):
        self.max_retry = 3
        self.logger = create_logger("Client")

        self.ssh = self.setup_ssh_server()
        hot_start = self.isup_manager()
        if not hot_start:
            self.init_manager()

        self.manager_ssh_server, self.manager_client_port = make_tunnel(5000)
        self.manager = self.get_manager()
        self.manager.is_alive()
        self.logger.debug("Successfully connected to Qsub manager on local port {0}".format(self.manager_client_port))

        if restart_manager and hot_start:
            self.restart_manager()

    def generator(self, package, module, wallclock, resources, rel_dir="", additional_modules=None):
        return QsubGenerator(self.manager, package, module, wallclock, resources, rel_dir, additional_modules)

    @staticmethod
    def setup_ssh_server():
        s = open_ssh_session_to_server()
        s.sendline('cd {0}'.format(WORKDIR))
        s.prompt()
        return s

    def isup_manager(self):
        self.ssh.sendline("python QsubTools.py isup manager".format(WORKDIR, SERVER_PYTHON_BIN))
        self.ssh.prompt()
        msg = self.ssh.before
        if "False" in msg:
            self.logger.debug("Manager down")
            return False
        elif "True" in msg:
            self.logger.debug("Manager up")
            return True
        else:
            raise Exception(msg)

    def init_manager(self, retries=0):
        timeout = retries * 2
        self.ssh.sendline("nohup python QsubTools.py init manager &")
        self.ssh.prompt()
        sleep(timeout)
        if not self.isup_manager():
            self.logger.info("Manager still not up after init")
            if retries > self.max_retry:
                self.logger.error("Giving up on initializing manager after {0} tries".format(self.max_retry))
            else:
                self.init_manager(retries=retries + 1)

    def get_manager(self):
        return Pyro4.Proxy("PYRO:qsub.manager@localhost:{0}".format(self.manager_client_port))

    def restart_manager(self):
        self.manager.shutdown()
        sleep(Pyro4.config.COMMTIMEOUT)
        self.init_manager()
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
        return self.latest_sub_id

    def get_state(self, sub_id):
        return self.qsubs[sub_id]['state']

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
        sub_id = self.manager.request_submission()
        manager = self.manager
        ss = self.submission_script
        class QsubInstance(BaseQsubInstance):
            @staticmethod
            def set_manager():
                return manager

            @staticmethod
            def set_sub_id():
                return sub_id

            @staticmethod
            def set_submission_script():
                return ss

        return QsubInstance


class SubmissionScript(object):
    def __init__(self, work_dir, modules):
        self.lines = ["#!/bin/sh"]
        self.wd = work_dir
        self.modules = modules

    def generate(self, execute_commands, log_file):
        script = '\n'.join(self.lines) + '\n'
        script += self.make_PBS('e', log_file + ".e") + '\n'
        script += self.make_PBS('o', log_file + ".o") + '\n'

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

    def make_PBS(self, flag, line):
        assert isinstance(line, (str, unicode))
        return "#PBS -" + flag.strip(' ') + ' ' + line

    def append_PBS(self, flag, line):
        self.lines.append(self.make_PBS(flag, line))


    def name(self, name):
        self.append_PBS('N ', name)

    def mail(self, mail_address):
        self.append_PBS('m', mail_address)

    def resources(self, resources):
        assert isinstance(resources, HPC_resources)
        self.append_PBS('l', 'nodes={1}:ppn={0}'.format(resources.ppn, resources.nodes))

        if resources.gpus:
            self.append_PBS('l', 'gpus={0}'.format(resources.gpus))

        if resources.pvmem:
            self.append_PBS('l', 'pvmem={0}'.format(resources.pvmem))

        if resources.vmem:
            self.append_PBS('l', 'vmem={0}'.format(resources.vmem))

    def wallclock(self, wallclock):
        assert isinstance(wallclock, HPC_Time)
        self.append_PBS("l", "walltime={0}:{1}:{2}".format(wallclock.h, wallclock.m, wallclock.s))


class BaseQsubInstance(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.sub_id = self.set_sub_id()
        self.qsub_manager = self.set_manager()
        self.submission_script = self.set_submission_script()

    @staticmethod
    def set_sub_id():
        return -1

    @staticmethod
    def set_manager():
        return Pyro4.Proxy("")

    @staticmethod
    def set_submission_script():
        return SubmissionScript(WORKDIR, BASE_MODULES)

class QsubExecutor(object):
    def __init__(self, cls, sub_id, manager_ip):
        self.manager = Pyro4.Proxy("PYRO:qsub.manager@{0}:5000")





if __name__ == "__main__":
    parameters = sys.argv[1:]
    print parameters
    if parameters[0] == 'init':
        if parameters[1] == 'manager':
            init_manager()
    elif parameters[0] == 'isup':
        if parameters[1] == 'manager':
            isup_manager()