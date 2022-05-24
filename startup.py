import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from jindai import *
from jindai.__main__ import _init_plugins as init
from PyMongoWrapper import F, Fn, Var
from bson import SON, Binary, ObjectId
import sys
import glob
