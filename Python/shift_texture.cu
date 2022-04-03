//  BSD 2-Clause License
//
//  Copyright (c) 2022, Eijiro SHIBUSAWA
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this
//     list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cuda_fp16.h>

extern "C" __global__ void shiftTexture(
	float* output,
	cudaTextureObject_t tex,
	int width,
	int height,
	int numShift,
	int minShift
	)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexS = blockIdx.y * blockDim.y + threadIdx.y;
	const int indexX = indexXY % width;
	const int indexY = indexXY / width;

	if ((indexX >= width) || (indexY >= height) || (indexS >= numShift))
	{
		return;
	}
	const int indexOutput = indexX + indexY * width + width * height * indexS;
	output[indexOutput] =  tex2D<float>(tex, indexX - indexS - minShift, indexY);
}

extern "C" __global__ void shiftTextureHalf(
	half* output,
	cudaTextureObject_t tex,
	int width,
	int height,
	int numShift,
	int minShift)
{
	const int indexXY = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexS = blockIdx.y * blockDim.y + threadIdx.y;
	const int indexX = indexXY % width;
	const int indexY = indexXY / width;

	if ((indexX >= width) || (indexY >= height) || (indexS >= numShift))
	{
		return;
	}
	const int indexOutput = indexX + indexY * width + width * height * indexS;
	output[indexOutput] =  __float2half(tex2D<float>(tex, indexX - indexS - minShift, indexY));
}
