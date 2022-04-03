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

import torch
import torch.nn as nn
import torch.nn.functional as F

class zero_mean_absolute_difference(nn.Module):
    def __init__(self, block_size = (5, 5)):
        super(zero_mean_absolute_difference, self).__init__()
        self.block_size = block_size
        self.half_block_size = block_size[0]//2, block_size[1]//2
        self.box_filter = (lambda x: F.avg_pool2d(x, self.block_size, stride=(1,1), padding=self.half_block_size, ceil_mode=False, count_include_pad=True))

    def forward(self, x, y):
        x2 = x - self.box_filter(x)
        y2 = y - self.box_filter(y)
        ad = torch.abs(y2 - x2)
        CV = self.box_filter(ad)
        return torch.argmin(CV[0], dim = 0)

if __name__ == '__main__':
    import PIL.Image as Image
    import numpy as np
    from stereo_shift import StereoShift
    from stereo_shift import write_debug_shift

    left = np.array(Image.open('data/teddy/im2.png').convert('L'))
    right = np.array(Image.open('data/teddy/im6.png').convert('L'))
    disparity_range = (1, 63)
    precision = np.float32
    assert(precision == np.float32 or precision == np.float16)
    t_precision = torch.float16 if precision == np.float16 else torch.float32
    device = 'cuda'
    enable_write_debug_shift = False
    enable_write_onnx = True

    t_left = torch.from_numpy(np.divide(np.array(left), 255, dtype=precision, casting='safe')).to(device)
    t_shift = torch.empty((disparity_range[1] - disparity_range[0], *left.shape), dtype=t_precision, device=device)

    ss = StereoShift(right, precision)
    ss.get(t_shift.data_ptr(), disparity_range)

    if enable_write_debug_shift:
        right_shift = t_shift.to('cpu').numpy()
        write_debug_shift(left, right_shift)

    zsad = zero_mean_absolute_difference()
    with torch.no_grad():
        t_indices = zsad.forward(t_left[None, None, :], t_shift[None, :])
        indices = t_indices.to('cpu').numpy()
        zsad_img = Image.fromarray((4 * (indices + disparity_range[0])).astype(np.uint8))
        zsad_img.save('zsad_disparity.png')

    if enable_write_onnx:
        dummy_x=torch.randn(1, 1, *t_left.shape).to('cuda')
        dummy_y=torch.randn(1, *t_shift.shape).to('cuda')
        torch.onnx.export(zsad, (dummy_x, dummy_y), "zsad.onnx", verbose=False,
            input_names=["x", "y"],
            output_names=["minimum_indices"],
            dynamic_axes={"x" : {2 : 'image_height', 3 : 'image_width'},
                          "y" : {1: 'disparity_size', 2 : 'image_height', 3 : 'image_width'},
                          "minimum_indices" : {0 : 'image_height', 1 : 'image_width'}})
