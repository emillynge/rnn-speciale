__author__ = 'emil'
USER = "s082768@student.dtu.dk"
WORKDIR = "/zhome/25/2/51526/Speciale/rnn-speciale"
LOGDIR = "/zhome/25/2/51526/Speciale/rnn-speciale/qsub/logs"
SERVER_PYTHON_BIN = "/usr/local/gbar/apps/python/2.7.1/bin/python"
QSUBMANAGER_PORT = 5000
SSH_USERNAME = "s082768"
SSH_PRIVATE_KEY = "/home/emil/.ssh/id_rsa"
SSH_HOST = "hpc-fe1.gbar.dtu.dk"
SSH_PORT = 22

BASEMODULES = {"python": '', "cuda": '', "boost": ''}
from subprocess import Popen, PIPE
FNULL = open('/dev/null', 'w')
import re
import os
import Pyro4.socketutil
from Pyro4 import errors as pyro_errors
import logging
import sys
from time import sleep

Pyro4.config.COMMTIMEOUT = 5.0 # without this daemon.close() hangs

from collections import namedtuple
class InvalidQsubArguments(Exception):
    pass

HPC_Time = namedtuple("Time", ['h', 'm', 's'])
HPC_resources = namedtuple("Resource", ['nodes', 'ppn', 'gpus', 'vmem'])

def create_logger():
    # create logger with 'spam_application'
    logger = logging.getLogger('Qsub')
    logger.setLevel(logging.DEBUG)
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
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

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

class QsubClient(object):
    def __init__(self):
        self.logger = logger
        self.ssh = self.setup_ssh_server()
        self.max_retry = 5

        if not self.isup_manager():
            self.init_manager()

        self.manager_ssh_server, self.manager_client_port = make_tunnel(5000)
        self.manager = self.get_manager()

    def setup_ssh_server(self):
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USERNAME, key_filename=SSH_PRIVATE_KEY)
        return ssh

    def isup_manager(self):
        (i, o, e) = self.ssh.exec_command("cd  %s; %s QsubTools.py isup manager" % (WORKDIR, SERVER_PYTHON_BIN))
        msg = o.readlines()
        self.logger.debug(msg)
        if "False\n" in msg:
            self.logger.debug("Manager down")
            return False
        elif "False\n" in msg:
            self.logger.debug("Manager up")
            return True
        else:
            raise  Exception(e.readlines())

    def init_manager(self, retries=0):
        self.ssh.exec_command("cd  %s; %s QsubTools.py init manager" % (WORKDIR, SERVER_PYTHON_BIN), timeout=4)
        if not self.isup_manager():
            self.logger.info("Manager still not up after init")
            if retries > self.max_retry:
                self.logger.error("Giving up on initializing manager after %d tries" % self.max_retry)
            else:
                self.init_manager(retries=retries+1)

    def get_manager(self, retries=0):
        return Pyro4.Proxy("PYRO:qsub.manager@localhost:%d" % self.manager_client_port)

def init_manager():
    daemon = Pyro4.Daemon(port=QSUBMANAGER_PORT)
    manager = QsubManager()
    daemon.register(manager, "qsub.manager")
    print "putting manager in request loop"
    daemon.requestLoop(loopCondition=manager.is_alive)

def isup_manager():
    manager = Pyro4.Proxy("PYRO:qsub.manager@localhost:5000")
    try:
        if manager.is_alive():
            print True
    except pyro_errors.CommunicationError:
        print False

class QsubManager(object):
    def __init__(self):
        self.available_modules = self.get_available_modules()
        active_qsubs = dict()
        self.running = True

    @staticmethod
    def get_available_modules():
        p = Popen("module avail", stdout=FNULL, stderr=PIPE, shell=True)
        lines = re.findall('/apps/dcc/etc/Modules/modulefiles\W+(.+)',
                           p.communicate()[1], re.DOTALL)
        module_list = re.findall('([^ \t])/([^ \t])[ \t\(]', '\n'.join(lines))
        module_dict = dict()
        for (module, version) in module_list:
            if module not in module_dict:
                module_dict[module] = list()
            module_dict[module] = version
        return module_dict

    def is_alive(self):
        return self.running

    def shutdown(self):
        print 'shutting down qsub manager'
        self.running = False


class QsubGenerator(object):
    def __init__(self, jobname, available_modules, hours=0, minutes=0, seconds=0, nodes=1, processors_per_node=1, gpus=0,
                 memory_per_process=None, memory_total=None, base_dir=WORKDIR, rel_dir=None, additional_modules={}):
        self.available_modules = available_modules
        if not any([hours, minutes, seconds]):
            raise InvalidQsubArguments('No wall clock time assigned to job: %d:%d:%d' % (hours, minutes, seconds))

        if nodes < 1 or processors_per_node < 1:
            raise InvalidQsubArguments('A job must have at least 1 node and 1 processor')

        self.resources = HPC_resources(nodes=nodes, ppn=processors_per_node, gpus=gpus, pvmem=memory_per_process,
                                       vmem=memory_total)
        self.wc_time = HPC_Time(h=hours, m=minutes, s=seconds)
        self.base_dir = base_dir
        self.rel_dir = rel_dir
        if not os.path.exists(self.work_dir):
            raise InvalidQsubArguments("Work directory %s doesn't exist." % self.work_dir)

        self.modules = BASEMODULES
        self.modules.update(additional_modules)
        self.check_modules()

    @property
    def work_dir(self):
        return self.base_dir + '/' + self.rel_dir


    def check_modules(self):
        for (module, version) in self.modules.iteritems():
            if module not in self.available_modules:
                raise InvalidQsubArguments("Required module %s is not available" % module)
            if version and version not in self.available_modules[module]:
                raise InvalidQsubArguments("Required module version %s is not available for module %" % (version, module))

    def write_submit(self):
        pass

    def submit(self):
        pass

    def read_logs(self):
        pass

if __name__ == "__main__":
    parameters  = sys.argv[1:]
    print parameters
    if parameters[0] == 'init':
        if parameters[1] == 'manager':
            init_manager()
    elif parameters[0] == 'isup':
        if parameters[1] == 'manager':
            isup_manager()