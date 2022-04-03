# BSD 2-Clause License
#
# Copyright (c) 2022, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import time

import tensorrt as trt

import numpy as np
import cupy as cp
import PIL.Image as Image

from stereo_shift import StereoShift

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_image', type=str, default='./data/teddy/im2.png')
    parser.add_argument('--right_image', type=str, default='./data/teddy/im6.png')
    parser.add_argument('--output_dir', type=str, default='/tmp')
    parser.add_argument('--engine_file', type=str, default='./zsad_engine.trt')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # setup left and right image
    left = np.array(Image.open(args.left_image).convert('L'))
    right = np.array(Image.open(args.right_image).convert('L'))
    disparity_range = (1, 63)

    # setup engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(args.engine_file, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # get shape
    left_shape = engine.get_profile_shape(0, engine[0])[1] # index 1 is 'optimal' shape
    shift_shape = engine.get_profile_shape(0, engine[1])[1]
    input_dtype_np = trt.nptype(engine.get_binding_dtype(engine[0]))
    output_dtype_np = trt.nptype(engine.get_binding_dtype(engine[2]))
    assert(left_shape[2:] == left.shape)
    assert(shift_shape[2:] == right.shape)
    assert(shift_shape[1] == disparity_range[1] - disparity_range[0])
    # allocate memory
    d_left = cp.array(np.divide(left, 255, dtype=input_dtype_np, casting='safe'))
    d_shift = cp.empty(shift_shape[1:], dtype=input_dtype_np)
    d_indices = cp.empty(left.shape, dtype=output_dtype_np)
    ss = StereoShift(right, d_shift.dtype)
    ss.get(d_shift.data, disparity_range)

    bindings = [int(d_left.data), int(d_shift.data), int(d_indices.data)]
    stream = cp.cuda.get_current_stream()
    context.execute_async_v2(bindings, stream.ptr, None)
    stream.synchronize()

    indices = cp.asnumpy(d_indices)
    zsad_img = Image.fromarray((4 * (indices + disparity_range[0])).astype(np.uint8))
    zsad_img.save(os.path.join(args.output_dir, 'zsad_disparity.png'))
