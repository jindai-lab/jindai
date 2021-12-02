#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import ctypes
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

sofile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfacedetection.so")
dll = ctypes.CDLL(sofile, mode=ctypes.RTLD_GLOBAL)

# THE SIGNATURE OF facedetect_cnn
# unsigned char * result_buffer
# unsigned char * rgb_image_data
# int width, int height, int step
cnnfunc = dll.facedetect_cnn
cnnfunc.restype = ctypes.POINTER(ctypes.c_int)
cnnfunc.argtypes = [ctypes.POINTER(ctypes.c_ubyte),
                               ctypes.POINTER(ctypes.c_ubyte),
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Faces: [(x y width height confidence)]
Faces = List[Tuple[int, int, int, int]]
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
c_short_p = ctypes.POINTER(ctypes.c_short)
c_int_p = ctypes.POINTER(ctypes.c_int)


# noinspection PyTypeChecker
# noinspection PyUnresolvedReferences
# noinspection PyCallingNonCallable
def cfacedetect_cnn(image: bytes,
                    width: int,
                    height: int,
                    step: int) -> Faces:
    image_type = ctypes.c_ubyte * (step * height)
    image_data = image_type.from_buffer_copy(image)
    # the magic number 0x20000 is DETECT_BUFFER_SIZE
    # which used in the original example source code
    result_buffer = (ctypes.c_ubyte * 0x20000)()
    # the return value is (int *)(result_buffer)
    # so just ignore it and use the result_buffer
    cnnfunc(result_buffer, image_data,
                       width, height, step)
    length = ctypes.cast(result_buffer, c_int_p)[0]
    faces = ctypes.cast(result_buffer, c_short_p)
    faces_results = []
    for i in range(length):
        start_addr = 2 + 142 * i
        c = faces[start_addr]
        x = faces[start_addr + 1]
        y = faces[start_addr + 2]
        w = faces[start_addr + 3]
        h = faces[start_addr + 4]
        result = (x, y, w, h, c)
        faces_results.append(result)
    # since here is Python
    # no more length needed
    return faces_results


# noinspection PyTypeChecker
def facedetect_cnn(image: Image,
                   width: int = 0, height: int = 0, step: int = 0) -> Faces:
    # if there is size given
    # then resize the image
    if width and height:
        size = (width, height)
        image = image.resize(size)
    image = image.convert('RGB')
    image = np.array(image)
    image = image[..., ::-1]
    
    width = image.shape[1]
    height = image.shape[0]
    depth = image.shape[2]
    step = width * depth
    image = image.tobytes()
    
    return cfacedetect_cnn(image, width, height, step)
