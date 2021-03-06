# Copyright 2021 Oscar José Pellicer Valero
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np

class RandomChannelDeleteTransform(AbstractTransform):
    '''
        Randomly selects channels to drop with probability p.

        Arguments
        ---------
            channels: list of lists of ints
                List of list of channels. E.g.: [[1], [2,3]] means to drop channel 1
                with probability= `p` and/or channels 2,3 (at the same time) with probability= `p`
            p: float, default 0.05
                Probability of dropping any given channel
    '''

    def __init__(self, channels, p=0.05, data_key="data"):
        self.data_key = data_key
        self.channels = channels
        self.p= p

    def __call__(self, **data_dict):
        for i in range(data_dict[self.data_key].shape[0]):
            #Get channels to drop and flatten them
            channels_to_drop= sum([ c for c in self.channels if np.random.random() < self.p ], [])
            data_dict[self.data_key][i,channels_to_drop] = 0.
        return data_dict
