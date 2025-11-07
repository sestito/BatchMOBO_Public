# Do the below for every file.


from .ZDT1 import ZDT1
from .ZDT2 import ZDT2
from .ZDT3 import ZDT3
from .FON import FON
from .DTLZ1 import DTLZ1
from .DTLZ2 import DTLZ2



'''
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
'''

'''
https://link.springer.com/content/pdf/10.1007%2F1-84628-137-7_6.pdf
https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.dtlz1
https://sop.tik.ee.ethz.ch/download/supplementary/testproblems/dtlz2/index.php
http://jmetal.sourceforge.net/problems.html
'''