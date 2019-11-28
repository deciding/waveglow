# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
#import os
from scipy.io.wavfile import write
import torch
import sys
sys.path.append('waveglow')
from .denoiser import Denoiser
MAX_WAV_VALUE = 32768.0
#python3 inference.py -f <(ls infer/mel_spectrograms/*.pt) -w infer/waveglow_256channels_ljs_v3.pt -o infer/ --is_fp16 -s 0.6

class WaveglowSynthesizer:
    def __init__(self,
            waveglow_path='/home/zining/workspace/waveglow/infer/waveglow_256channels_ljs_v3.pt',
            sigma=0.6, sampling_rate=22050, is_fp16=True, denoiser_strength=0.0):
        waveglow = torch.load(waveglow_path)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.cuda().eval()
        if is_fp16:
            from apex import amp
            waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

        if denoiser_strength > 0:
            self.denoiser = Denoiser(waveglow).cuda()
        self.waveglow=waveglow
        self.sigma=sigma
        self.sampling_rate=sampling_rate
        self.is_fp16=is_fp16
        self.denoiser_strength=denoiser_strength

    def synthesize(self, mel, output_path):
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if self.is_fp16 else mel
        with torch.no_grad():
            audio = self.waveglow.infer(mel, sigma=self.sigma)
            if self.denoiser_strength > 0:
                audio = self.denoiser(audio, self.denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = output_path
        write(audio_path, self.sampling_rate, audio)
        #print(audio_path)

