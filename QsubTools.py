import os
import datetime

__author__ = 'emil'
USER = "s082768@student.dtu.dk"
WORKDIR = "/zhome/25/2/51526/Speciale/rnn-speciale"
SITE_PACKAGE_DIR = "/zhome/25/2/51526/.local/lib/python2.7/site-packages"
LOGDIR = "/zhome/25/2/51526/Speciale/rnn-speciale/qsubs/logs"
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
from Pyro4 import errors as pyro_errors, Daemon
import logging
from boltons import tbutils
import traceback
import sys
import json
from copy import copy
from time import sleep
from importlib import import_module
from functools import partial
Pyro4.config.COMMTIMEOUT = 5.0  # without this daemon.close() hangs

from collections import namedtuple, defaultdict


class InvalidUserInput(Exception):
    def __init__(self, message, argnames=[], expected='', found='', should='', indent=3, **kwargs):
        if isinstance(argnames, str):
            argnames = [argnames]
        calling_frame = sys._getframe(indent)
        method_frame = sys._getframe(indent - 1)

        tb = tbutils.TracebackInfo.from_frame(frame=calling_frame, limit=1)
        tb_str = tb.get_formatted()
        declaration, input_args, method_name = self.get_declaration(method_frame)

        self.data = {'expected': expected,
                     'found': found,
                     'argnames': argnames,
                     'method_name': method_name,
                     'pre_message': tb_str[34:] if tb_str else "",
                     'message': message,
                     'declaration': declaration,
                     'input_args': input_args,
                     'arg_no': [input_args.index(argname) + 1 for argname in argnames if argname in input_args],
                     'should': should}

        super(InvalidUserInput, self).__init__(self.message, **kwargs)

    def get_declaration(self, frame):
        lno = frame.f_code.co_firstlineno
        with open(frame.f_code.co_filename) as fp:
            for k in range(lno-1):
                fp.readline()

            declaration_buffer = fp.readline()
            while '):' not in declaration_buffer:
                declaration_buffer += fp.readline()
            input_str = re.findall('def\W+{0}\((.+)\):'.format(frame.f_code.co_name),
                                   declaration_buffer, re.DOTALL)[0]
            declaration_str = re.findall('def\W+({0}\(.+?\)):'.format(frame.f_code.co_name), declaration_buffer)[0]

        return declaration_str, self.parse_input_str(input_str), frame.f_code.co_name

    @property
    def super_message(self):
        return super(InvalidUserInput, self).message

    @property
    def message(self):
        message = "Invalid input to {declaration}{pre_message}\t".format(**self.data)
        if self.data['argnames']:
            message += 'argument {arg_no} "{argnames}"'.format(**self.data)

        if self.data['found']:
            message += ' was "{found}"'.format(**self.data)

        if self.data['expected']:
            message += ' but {method_name} requires {should} "{expected}"'.format(**self.data)

        if self.data['message']:
            message += '\n\t' + self.data['message']
        return message

    @staticmethod
    def compare(argname, expect, found, message="", equal=True):
        if equal and expect != found:
            raise InvalidUserInput(message, argname, expect, found, should='it to be')
        elif not equal and expect == found:
            raise InvalidUserInput(message, argname, expect, found, should='it not to be')

    @staticmethod
    def parse_input_str(input_str):
        quotes = defaultdict(int)
        input_args = list()
        curr_arg = ""
        for char in input_str:
            if char in '([{':
                quotes[char] += 1
                continue

            if char in ')]}':
                quotes[char] -= 1
                if quotes[char] == 0:
                    del(quotes[char])
                continue

            if char in '\n\r\t':
                continue

            if char == ',' and not quotes:
                input_args.append(curr_arg)
                curr_arg = ''
                continue

            curr_arg += char
        input_args.append(curr_arg)
        input_args = [a.strip(' ') for a in input_args]
        if 'self' in input_args:
            input_args.remove('self')
        return input_args


class InvalidQsubArguments(InvalidUserInput):
    pass


def import_obj_from(module_name, obj_name):
    return import_module(module_name).__dict__[obj_name]


def open_ssh_session_to_server():
    import pxssh

    s = pxssh.pxssh()
    if not s.login(SSH_HOST, SSH_USERNAME, ssh_key=SSH_PRIVATE_KEY):
        raise pxssh.ExceptionPxssh('Login failed')
    return s

def make_path(path, ignore_last=False):
    paths = path.split('/')
    if ignore_last:
        paths = paths[:-1]

    abs_path = paths[0]
    exists = True
    for p in paths[1:] + ['']:
        if not exists or not os.path.exists(abs_path):
            os.mkdir(abs_path)
            exists = False
        abs_path += '/' + p
        if not p:
            break


def create_logger(logger_name="Qsub", log_to_file='logs/qsubs.log', log_to_stream=True, log_level='DEBUG',
                  format_str=None):
    if not log_to_file and not log_to_stream:  # neither stream nor logfile specified. no logger wanted.
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
            make_path(logfile, ignore_last=True)
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
ClassInfo = namedtuple('ClassInfo', ['module', 'class_name'])


class QsubClient(object):
    def __init__(self, restart_manager=False):
        self.max_retry = 3
        self.logger = create_logger("Client")

        self.manager_ip, hot_start = RemoteQsubCommandline('-i isup manager').get('ip', 'return')
        if not hot_start:
            self.start_manager()

        self.manager_ssh_server, self.manager_client_port = make_tunnel(5000, server_host=self.manager_ip)
        self.manager = self.get_manager()
        if not self.manager.is_alive():
            self.logger.error('could not start manager')
            raise Exception("Could not start manager")
        self.logger.debug("Successfully connected to Qsub manager on  {1}:{0}".format(self.manager_client_port,
                                                                                      self.manager_ip))

        if restart_manager and hot_start:
            self.restart_manager()

    def instance_generator(self, cls, wallclock, resources, rel_dir="", additional_modules=None):
        if not isinstance(cls, tuple):
            InvalidUserInput.compare('cls', "<type 'type'>", str(cls.__class__),
                                     message="Input should be uninstantiated class")
            cls = ClassInfo(module=cls.__module__, class_name=cls.__name__)

        qsub_gen = QsubGenerator(self, cls, wallclock, resources, rel_dir, additional_modules)
        return qsub_gen

    def start_manager(self):
        self.manager_ip = RemoteQsubCommandline('-i start manager').get('ip')
        # timeout = retries * 2
        # RemoteQsubCommandline('init manager')
        # sleep(timeout)
        # if not self.isup_manager():
        # self.logger.info("Manager still not up after init")
        #     if retries > self.max_retry:
        #         self.logger.error("Giving up on initializing manager after {0} tries".format(self.max_retry))
        #     else:
        #         self.start_manager(retries=retries + 1)

    def get_manager(self):
        return QsubProxy("PYRO:qsub.manager@localhost:{0}".format(self.manager_client_port, self.manager_ip))

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
    def __init__(self, logger=None):
        self.logger = logger or create_logger('Manager')
        self._available_modules = self.get_available_modules()
        self.running = True
        self.latest_sub_id = -1
        self.qsubs = defaultdict(dict)
        self.ip = self.get_ip()
        self.logger.info("Manager started on {0}".format(self.ip))

    def available_modules(self):
        return self._available_modules

    def request_submission(self):
        self.latest_sub_id += 1
        self.qsubs[self.latest_sub_id]['state'] = 'requested'
        self.logger.info("Submission id: {0} granted".format(self.latest_sub_id))
        return self.latest_sub_id, self.logfile(self.latest_sub_id)

    def stage_submission(self, sub_id, script):
        with open(self.subid2sh(sub_id), 'w') as fp:
            fp.write(script)
        self.qsubs[sub_id]['state'] = 'staged'

    def submit(self, sub_id, args, kwargs):
        self.qsubs[sub_id]['args'] = args
        self.qsubs[sub_id]['kwargs'] = kwargs
        p_sub = Popen(['qsub', self.subid2sh(sub_id)], stdout=PIPE, stderr=PIPE)
        stdout = p_sub.stdout.read()
        job_id = re.findall('(\d+)\.\w+', stdout)[0]
        self.qsubs[sub_id]['job_id'] = job_id
        return self.qstat(sub_id)

    def request_execution_args(self, sub_id):
        self.qsubs[sub_id]['state'] = 'init'
        return self.qsubs[sub_id]['args'], self.qsubs[sub_id]['kwargs']

    def set_proxy_info(self, sub_id, daemon_ip, daemon_port, proxy_name):
        self.qsubs[sub_id]['proxy_info'] = {'ip': daemon_ip, 'port': daemon_port, 'name': proxy_name}
        self.qsubs[sub_id]['state'] = 'ready'

    def get_proxy_info(self, sub_id):
        if self.qsubs[sub_id]['state'] == 'ready':
            return self.qsubs[sub_id]['proxy_info']
        else:
            return None

    def qstat(self, sub_id):
        p_stat = Popen(['qstat', self.qsubs[sub_id]['job_id']], stdout=PIPE)
        vals = re.split('[ ]+', re.findall(self.qsubs[sub_id]['job_id'] + '.+', p_stat.stdout.read())[0])
        keys = ['Job ID', 'Name', 'User', 'Time Use', 'S', 'Queue']
        state = dict(zip(keys, vals[:-1]))

        def time_str2time_sec(time_str):
            time_tup = re.findall('(\d+):(\d+):(\d+)', time_str)
            if time_tup:
                return int(time_tup[0][0]) * 60 * 60 + int(time_tup[0][1]) * 60 + int(time_tup[0][0])
            return -1

        if state['S'] == 'Q':
            self.qsubs[sub_id]['state'] = 'queued'
            p_start = Popen(['showstart',  self.qsubs[sub_id]['job_id']], stdout=PIPE)
            time_str = re.findall('Estimated Rsv based start in\W+(\d+:\d+:\d+)', p_start.stdout.read())
            if time_str:
                time_sec = time_str2time_sec(time_str[0])
            else:
                time_sec = -1
            return 'queued', time_sec

        if state['S'] == 'R':
            if self.qsubs[sub_id]['state'] not in ['running', 'init', 'ready']:
                self.qsubs[sub_id]['state'] = 'running'
            return self.qsubs[sub_id]['state'], time_str2time_sec(state['Time Use'])

        elif state['S'] == 'C':
            self.qsubs[sub_id]['state'] = 'completed'
            return 'completed', time_str2time_sec(state['Time Use'])

    def subid2sh(self, sub_id):
        return 'qsubs/{0}.sh'.format(sub_id)

    def get_state(self, sub_id):
        return self.qsubs[sub_id]['state']

    @staticmethod
    def logfile(sub_id):
        return '{0}/{1}'.format(LOGDIR, sub_id)

    def error_log(self, sub_id):
        return self.logfile(sub_id) + '.e'

    def out_log(self, sub_id):
        return self.logfile(sub_id) + '.o'

    def get_available_modules(self):
        p = Popen("module avail", stdout=FNULL, stderr=PIPE, shell=True)
        (o, e) = p.communicate()
        if o:
            lines = re.findall('/apps/dcc/etc/Modules/modulefiles\W+(.+)',
                               o[1], re.DOTALL)
        else:
            lines = list()

        if lines:
            lines = lines[0]
        else:
            self.logger.error('module avail command failed: {0}'.format(e))
            return dict()

        self.logger.debug(lines)
        modules = re.split('[ \t\n]+', lines)[:-1]
        self.logger.debug(modules)
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
    def __init__(self, qsub_client, cls, wallclock, resources, rel_dir, additional_modules):

        assert isinstance(wallclock, HPC_Time)
        assert isinstance(resources, HPC_resources)
        assert isinstance(qsub_client, QsubClient)
        qsub_manager = qsub_client.manager

        self.qsub_client = qsub_client
        self.available_modules = qsub_manager.available_modules()
        self.resources = None
        self.wc_time = None
        self.modules = copy(BASE_MODULES)
        self.base_dir = WORKDIR
        self.rel_dir = rel_dir
        self.cls = cls
        self.resources = resources
        self.wc_time = wallclock
        self.logger = create_logger('Generator')
        self.manager = qsub_manager

        try:
            if resources.nodes < 1 or resources.ppn < 1:
                raise InvalidQsubArguments('A job must have at least 1 node and 1 processor', argnames='resources',
                                           found=resources)

            if not any(wallclock):
                raise InvalidQsubArguments('No wall clock time assigned to job', argnames='wallclock', found=wallclock,
                                           )

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
        ss.name('.'.join(self.cls))
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
        qsub_client = self.qsub_client

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

            @staticmethod
            def set_qsub_client():
                return qsub_client

        return QsubInstance


class SubmissionScript(object):
    def __init__(self, work_dir, modules):
        self.lines = ["#!/bin/sh"]
        self.wd = work_dir
        self.modules = modules

    def generate(self, execute_commands, log_file):
        script = self.make_pbs_pragma('e', log_file + ".e") + '\n'
        script += self.make_pbs_pragma('o', log_file + ".o") + '\n'
        script += '\n'.join(self.lines) + '\n'
        for module_name, version in self.modules.iteritems():
            script += 'module load ' + module_name
            if version:
                script += '/' + version
            script += '\n'
        script += "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{0}/lib\n".format(WORKDIR)
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
        self.append_pbs_pragma('M', mail_address)
        self.append_pbs_pragma('m', 'a')

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
        self.remote_controller = None
        self.remote_obj = None
        self.object_ssh_server = None
        self.object_client_port = None
        self.proxy_info = None
        self.stage_submission()

    def stage_submission(self):
        kwargs = {'manager_ip': self.qsub_client.manager_ip,
                  'cls': self.qsub_generator.cls.class_name,
                  'module': self.qsub_generator.cls.module,
                  'sub_id': self.sub_id}
        exe = "python QsubTools.py start executor manager_ip={manager_ip} cls={cls} module={module} sub_id={sub_id}".format(**kwargs)
        script = self.submission_script.generate(exe, self.logfile)
        self.qsub_manager.stage_submission(self.sub_id, script)

    def __enter__(self):
        state, t = self.qsub_manager.submit(self.sub_id, self.args, self.kwargs)
        while state != 'ready':
            if t > 0:
                self.qsub_client.logger.debug('Waiting for remote object.\n\t State: {0}\n\t Seconds left: {1}'.format(state, t))
                sleep(min([t, 30]))
            state, t = self.qsub_manager.qstate(self.sub_id, self.args, self.kwargs)

        self.proxy_info = self.qsub_manager.get_proxy_info(self.sub_id)
        self.object_ssh_server, self.object_client_port = make_tunnel(self.proxy_info['port'],
                                                                      server_host=self.proxy_info['ip'])

        self.remote_controller = Pyro4.Proxy('PYRO:qsub.execution.controller@localhost:{1}'.format(self.proxy_info['name'],
                                                                             self.object_client_port))
        self.remote_obj = Pyro4.Proxy('PYRO:{0}@localhost:{1}'.format(self.proxy_info['name'],
                                                                             self.object_client_port))
        return self.remote_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.remote_controller:
            self.remote_controller.shutdown()
        if self.object_ssh_server:
            self.object_ssh_server.stop()

    def __del__(self):
        self.__exit__(1, 2, 3)

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

class ExecutionController(object):
    def __init__(self, logger):
        self.running = True
        self.logger = logger

    def is_alive(self):
        if self.running:
            self.logger.debug('alive')
            return True
        else:
            self.logger.debug('dead')
            return False

    def shutdown(self):
        self.logger.info("shutdown signal received")
        self.running = False


def init_server_execution(module, cls, sub_id, manager_ip, local_ip, logger=None):
    logger = logger if logger else create_logger(logger_name="Executor", log_to_file=[])

    manager = Pyro4.Proxy("PYRO:qsub.manager@{0}:5000".format(manager_ip))
    if manager.is_alive():
        logger.info("manager found!")
    else:
        logger.error('no manager found')
        raise Exception('no manager found')

    args, kwargs = manager.request_execution_args(sub_id)
    obj = import_obj_from(module, cls)(*args, **kwargs)
    wrapped_object = ServerExecutionWrapper(obj)
    daemon = QsubDaemon(host=local_ip)
    port = daemon.locationStr.split(':')[-1]
    proxy_name = 'qsub.execution.{0}'.format(sub_id)
    daemon.register(wrapped_object, proxy_name)
    controller = ExecutionController(logger)
    daemon.register(controller, 'qsub.execution.controller')
    manager.set_proxy_info(sub_id, local_ip, port, proxy_name)
    daemon.requestLoop(controller.is_alive)

class QsubDaemon(Pyro4.Daemon):
    def __init__(self, *args, **kwargs):
        super(QsubDaemon, self).__init__(*args, **kwargs)

        def get_metadata(objectId):
            obj = self.objectsById.get(objectId)
            if obj is not None:
                if hasattr(obj, 'QSUB_metadata'):
                    return getattr(obj, 'QSUB_metadata')
                return Pyro4.util.get_exposed_members(obj, only_exposed=Pyro4.config.REQUIRE_EXPOSE)
            else:
                Pyro4.core.log.debug("unknown object requested: %s", objectId)
                raise Pyro4.errors.DaemonError("unknown object")

        setattr(self.objectsById['Pyro.Daemon'], 'get_metadata', get_metadata)


class ServerExecutionWrapper(object):
    def __init__(self, obj):
        methods = set()
        attrs = set()
        oneway = set()
        # exposing methods of wrapped object
        for method_name in obj.__class__.__dict__.keys():
            if method_name[0] != '_':
                setattr(self, method_name, getattr(obj, method_name))
                methods.add(method_name)

        # exposing properties of wrapped object
        for (prop_name, prop) in obj.__dict__.items():
            setattr(self, 'QSUB_fget_' + prop_name, partial(getattr, obj, prop_name))
            methods.add('QSUB_fget_' + prop_name)
            setattr(self, 'QSUB_fset_' + prop_name, partial(setattr, obj, prop_name))
            methods.add('QSUB_fset_' + prop_name)
            setattr(self, 'QSUB_fdel_' + prop_name, partial(delattr, obj, prop_name))
            methods.add('QSUB_fdel_' + prop_name)

        self.QSUB_metadata = {"methods": methods,
                              "oneway": oneway,
                              "attrs": attrs}

def QsubProxy(*args, **kwargs):
    proxy = Pyro4.Proxy(*args, **kwargs)
    return wrap_execution_proxy(proxy)

def wrap_execution_proxy(pyro_proxy):
    pyro_proxy._pyroGetMetadata()
    props = defaultdict(dict)
    in_props = defaultdict(dict)
    methods = dict()

    def manipulate_prop(action, propname, self, *args):
        return self._props[action][propname].__call__(*args)

    for method_name in pyro_proxy._pyroMethods:
        regexp = re.findall('^QSUB_((fget)|(fset)|(fdel))_(.+$)', method_name)
        if regexp:
            props[regexp[0][-1]][regexp[0][0]] = pyro_proxy.__getattr__(method_name)
            in_props[regexp[0][-1]][regexp[0][0]] = partial(manipulate_prop, regexp[0][-1], regexp[0][0])
        else:
            methods[method_name] = pyro_proxy.__getattr__(method_name)

    class ClientExecutionWrapper(object):
        def __init__(self, _props):
            self._props = _props

    for m_name, m in methods.items():
        setattr(ClientExecutionWrapper, m_name, m)

    for (prop_name, methods) in in_props.iteritems():
            setattr(ClientExecutionWrapper, prop_name, property(**methods))

    return ClientExecutionWrapper(props)


class QsubCommandline(object):
    def __init__(self, commands=None):
        self.args2method = {'manager': {'start': self.start_manager,
                                        'stop': self.stop_manager,
                                        'isup': self.isup_manager},
                            'executor': {'start': self.start_executor}}
        self.commands = commands
        self.data = dict()
        self.stdout_logger = create_logger('CLI/stdout', log_to_file="", log_to_stream=True, format_str='%(message)s')
        self.args = self.parse_args()
        if self.args.remote:
            self.data = RemoteQsubCommandline(' '.join([arg for arg in sys.argv[1:] if arg not in ['-r', '--remote']]))
        else:
            self.logger = self.create_logger()
            self.pre_execute()
            self.execute()
            self.post_execute()

    def get(self, *args):
        return tuple(self.data[key] for key in args)

    def parse_args(self):
        if self.commands is not None:
            args = self.get_argument_parser().parse_args(self.commands.split())
        else:
            args = self.get_argument_parser().parse_args()

        self.data['kwargs'] = dict()
        if args.kwargs:
            for kwarg in args.kwargs:
                if '=' not in kwarg:
                    raise self.CommandLineException('Invalid input "{0}": key-word arguments must contain a "="'.format(kwarg))
                self.data['kwargs'].update(dict([tuple(kwarg.split('='))]))
        return args

    def get_exec_func(self):
        if self.args.module in self.args2method and self.args.action in self.args2method[self.args.module]:
            return self.args2method[self.args.module][self.args.action]
        return self.method_not_implemented_func

    def method_not_implemented_func(self):
        self.stdout("error", "action {0} not implemented for module {1}".format(self.args.action, self.args.module))

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
            self.stdout('error', e.__class__.__name__ + ': ' + e.message)

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

    def start_executor(self):
        error_message = ""
        if 'manager_ip' not in self.data['kwargs']:
            error_message += '\n\tmanager_ip is missing.'
        if 'module' not in self.data['kwargs']:
            error_message += '\n\tmodule is missing.'
        if 'cls' not in self.data['kwargs']:
            error_message += '\n\tcls is missing.'
        if 'sub_id' not in self.data['kwargs']:
            error_message += '\n\tsub_id is missing.'

        if error_message:
            raise self.CommandLineException('start executor command requires key-word arguments.' + error_message)
        init_server_execution(self.data['kwargs']['module'], self.data['kwargs']['cls'], self.data['kwargs']['sub_id'],
                              self.data['kwargs']['manager_ip'], self.data['ip'], logger=self.logger)

    def start_manager(self):
        self.logger.debug("Initializing manager")
        daemon = QsubDaemon(port=QSUB_MANAGER_PORT, host=self.data['ip'])
        self.logger.debug("Init Manager")
        manager = ServerExecutionWrapper(QsubManager(logger=self.logger))
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
                            choices=['manager', 'executor'],
                            help="module to send action to")

        parser.add_argument('kwargs',
                            help="arguments to send to module",
                            nargs='*',
                            metavar='key=value')
        return parser

    @staticmethod
    class CommandLineException(Exception):
        pass


class RemoteQsubCommandline(QsubCommandline):
    def __init__(self, commands):
        self.ssh = self.setup_ssh_instance()
        self.ssh.ignore_sighup = False
        super(RemoteQsubCommandline, self).__init__(commands=commands)

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
        full_command += self.commands

        if blocking:
            full_command += ' > nohup.out & tail -f nohup.out'
        self.commands = full_command

    def execute(self):
        self.ssh.sendline(self.commands)

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
