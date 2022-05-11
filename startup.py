import os
import sys
import glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from PyMongoWrapper import F, Fn, Var
from jindai.__main__ import _init_plugins as init
from jindai import *