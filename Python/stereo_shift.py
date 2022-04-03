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

import math
import os

import numpy as np
import cupy as cp

class StereoShift():
    def __init__(self, img, dtype):
        dn = os.path.dirname(os.path.realpath(__file__))
        fpfn = os.path.join(dn, 'shift_texture.cu')
        # load raw kernel
        with open(fpfn, 'r') as f:
            cuda_source = f.read()
        self.module = cp.RawModule(code=cuda_source)
        if dtype == np.float32:
            self.copy_texture = self.module.get_function("shiftTexture")
        elif dtype == np.float16:
            self.copy_texture = self.module.get_function("shiftTextureHalf")

        # setup texture object
        channel_format_descriptor = cp.cuda.texture.ChannelFormatDescriptor(8, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindUnsigned)
        self.array = cp.cuda.texture.CUDAarray(channel_format_descriptor, img.shape[1], img.shape[0])
        self.array.copy_from(img)
        self.resouce_descriptor = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray,
            cuArr = self.array)
        self.texture_descriptor = cp.cuda.texture.TextureDescriptor(addressModes = (cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
            filterMode=cp.cuda.runtime.cudaFilterModePoint,
            readMode=cp.cuda.runtime.cudaReadModeNormalizedFloat,
            normalizedCoords = 0)
        self.texuture_object = cp.cuda.texture.TextureObject(self.resouce_descriptor, self.texture_descriptor)

    def get(self, ptr, disparity_range):
        num_disparity = disparity_range[1] - disparity_range[0]
        sz_block = 1024, 1
        sz_grid = math.ceil(self.array.width * self.array.height/ sz_block[0]), math.ceil(num_disparity / sz_block[1])
        # call the kernel
        self.copy_texture(
            block=sz_block,
            grid=sz_grid,
            args=(
                ptr,
                self.texuture_object,
                self.array.width,
                self.array.height,
                num_disparity,
                disparity_range[0]
            )
        )

if __name__ == '__main__':
    import PIL.Image as Image
    left = np.array(Image.open('data/teddy/im2.png').convert('L'))
    right = np.array(Image.open('data/teddy/im6.png').convert('L'))
    disparity_range = (1, 63)
    d_shift = cp.empty((disparity_range[1] - disparity_range[0], *left.shape), dtype=np.float16)

    ss = StereoShift(right, d_shift.dtype)
    ss.get(d_shift.data, disparity_range)

    right_shift = cp.asnumpy(d_shift)
    for shift, index in zip(right_shift, range(len(right_shift))):
        blend_img = Image.fromarray(((255 * shift + left)/2).astype(np.uint8))
        blend_img.save('shift_{0:2d}.png'.format(index))
