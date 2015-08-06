__author__ = 'emil'
from imp import load_source
from QsubTools import InvalidUserInput
class CUDA_tests(object):
    def __init__(self):
        self.test_classes = {'math': load_source('cumath', 'pycuda-git/test/test_cumath.py').TestMath,
                             'driver': load_source('driver', 'pycuda-git/test/test_driver.py').TestDriver,
                             'gpuarray': load_source('gpuarray', 'pycuda-git/test/test_gpuarray.py').TestGPUArray}

    def test_class(self, case):
        if case not in self.test_classes:
            raise InvalidUserInput('test case not found', 'case', self.test_classes.keys(), case)

        tests = [t for t in self.test_classes[case].__dict__.keys() if 'test' in t]
        test_obj = self.test_classes[case]()
        for test in tests:
            getattr(test_obj, test).__call__()
