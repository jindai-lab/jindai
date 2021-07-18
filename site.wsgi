import os, sys
fwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(fwd)
sys.path.append(fwd)
from parallel_corpus import app as application
