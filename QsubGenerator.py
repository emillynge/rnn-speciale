__author__ = 'emil'
USER = "s082768@student.dtu.dk"
WORKDIR = "/zhome/25/2/51526/Speciale/rnn-speciale"
LOGDIR = "/zhome/25/2/51526/Speciale/rnn-speciale/qsub/logs"
BASEMODULES = {"python": '', "cuda": '', "boost": ''}
from subprocess import Popen, PIPE
FNULL = open('/dev/null', 'w')
import re
import os
from collections import namedtuple
class InvalidQsubArguments(Exception):
    pass

HPC_Time = namedtuple("Time", ['h', 'm', 's'])
HPC_resources = namedtuple("Resource", ['nodes', 'ppn', 'gpus', 'vmem'])

class QsubGenerator(object):
    def __init__(self, jobname, hours=0, minutes=0, seconds=0, nodes=1, processors_per_node=1, gpus=0,
                 memory_per_process=None, memory_total=None, base_dir=WORKDIR, rel_dir=None, additional_modules={}):
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

    def available_modules(self):
        p = Popen("module avail", stdout=FNULL, stderr=PIPE, shell=True)
        lines = re.findall('/apps/dcc/etc/Modules/modulefiles\W+(.+)',
                           p.communicate()[1], re.DOTALL)
        module_list = re.findall('([^ \t])/([^ \t])[ \t\(]', lines)
        module_dict = dict()
        for (module, version) in module_list:
            if module not in module_dict:
                module_dict[module] = list()
            module_dict[module] = version
        return module_dict

    def check_modules(self):
        available_modules = self.available_modules()
        for (module, version) in self.modules.iteritems():
            if module not in available_modules:
                raise InvalidQsubArguments("Required module %s is not available" % module)
            if version and version not in available_modules[module]:
                raise InvalidQsubArguments("Required module version %s is not available for module %" % (version, module))


    def write_submit(self):
        pass

    def submit(self):
        pass

    def read_logs(self):
        pass








