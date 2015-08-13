__author__ = 'emil'
from QsubTools import RemoteQsubCommandline, BaseQsubInstance, QsubClient, QsubProxy, QsubManager, QsubCommandline, QSUB_MANAGER_PORT, ServerExecutionWrapper, QsubDaemon
from pexpect import spawn
from subprocess import Popen, PIPE
import re
import datetime
class pxsshTest(spawn):
    def __init__(self, *args, **kwargs):
        super(pxsshTest, self).__init__('bash -i')

    def logout(self):
        pass

class BaseQsubInstanceTest(object):
    def _get_execution_controller(self):
        self.execution_controller = QsubProxy('PYRO:qsub.execution.controller@{0}:{1}'.format(self.proxy_info['ip'],
                                                                                              self.object_client_port))

    def _get_obj(self, obj_info):
        return QsubProxy('PYRO:{0}@{2}:{1}'.format(obj_info['object_id'], self.object_client_port,
                         self.proxy_info['ip']))



class RemoteQsubCommandlineTest(RemoteQsubCommandline):
    """ Test version that replaces ssh with a local shell """
    @staticmethod
    def setup_ssh_instance():
        return pxsshTest()

    @property
    def interpreter(self):
        return 'python2'

    @property
    def target_file(self):
        return 'QsubTest.py'


class QsubClientTest(QsubClient):
    def start_manager(self):
        self.manager_ip = RemoteQsubCommandlineTest('-i start manager').get('ip')[0]

    def isup_manager(self):
        self.manager_ip, self.manager_running = RemoteQsubCommandlineTest('-i isup manager').get('ip', 'return')

    def make_tunnel(self):
        return None, 5000

    def get_manager(self):
        return QsubProxy("PYRO:qsub.manager@{1}:{0}".format(self.manager_client_port, self.manager_ip))

    @property
    def interpreter(self):
        return 'python2'

    @property
    def target_file(self):
        return 'QsubTest.py'

class QsubManagerTest(QsubManager):
    @property
    def logdir(self):
        return '/home/emil/Drive/Speciale/Kode/RNN/rnn-speciale/qsubs/logs'

    def _qsub(self, sub_id):
        p1 = Popen(['sh', self.subid2sh(sub_id)])
        p2 = Popen(['ps', '--ppid', str(p1.pid)], stdout=PIPE)
        p2.stdout.readline()
        line = p2.stdout.readline()
        job_id = re.findall('\d+', line)[0]
        return job_id

    def qdel(self, sub_id):
        self.qstat(sub_id)
        if not self.has_reached_state(sub_id, 'completed') and self.has_reached_state(sub_id, 'submitted'):
            Popen(['kill', '-9', self.qsubs[sub_id]['job_id']])

    def _qstat(self, sub_id):
        p_stat = Popen(['ps', '-q', self.qsubs[sub_id]['job_id']], stdout=PIPE)
        p_stat.stdout.readline()
        line = p_stat.stdout.readline()
        rexp = re.findall('(\d\d:\d\d:\d\d) (.+?)((<defunct>)|($))', line)
        vals = [self.qsubs[sub_id]['job_id'], 'bla', 'bla']
        if rexp:
            vals.append(rexp[0][0])
            if rexp[0][2]:
                vals.append('C')
            else:
                vals.append('R')
        else:
            vals.append('00:00:00')
            vals.append('C')

        vals.append('Test')

        keys = ['Job ID', 'Name', 'User', 'Time Use', 'S', 'Queue']
        state = dict(zip(keys, vals[:-1]))
        return state

class QsubCommandLineTest(QsubCommandline):
    def start_manager(self):
        self.logger.debug("Initializing manager")
        daemon = QsubDaemon(port=QSUB_MANAGER_PORT, host=self.data['ip'])
        self.logger.debug("Init Manager")
        manager = QsubManagerTest(logger=self.logger)
        wrapper = ServerExecutionWrapper(manager, logger=self.logger)
        daemon.register(wrapper, "qsub.manager")
        self.logger.info("putting manager in request loop")
        self.stdout('blocking', datetime.datetime.now().isoformat())
        daemon.requestLoop(loopCondition=manager.is_alive)

    def remote_call(self, commands):
        RemoteQsubCommandlineTest(' '.join([command for command in commands if commands not in ['-r', '--remote']]))

if __name__ == "__main__":
    QsubCommandLineTest()
