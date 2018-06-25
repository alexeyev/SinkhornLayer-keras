# Copyright 2018 Anton Alekseev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Layer


class SinkhornLayer(Layer):
    def __init__(self, n_iters=21, temperature=0.01, **kwargs):
        self.supports_masking = False
        self.n_iters = n_iters
        self.temperature = K.constant(temperature)
        super(SinkhornLayer, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        n = K.shape(input_tensor)[1]
        log_alpha = K.reshape(input_tensor, [-1, n, n])
        log_alpha /= self.temperature

        for _ in range(self.n_iters):
            log_alpha -= K.reshape(K.log(K.sum(K.exp(log_alpha), axis=2)), [-1, n, 1])
            log_alpha -= K.reshape(K.log(K.sum(K.exp(log_alpha), axis=1)), [-1, 1, n])
        return K.exp(log_alpha)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape
