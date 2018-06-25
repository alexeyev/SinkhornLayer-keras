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

import argparse

import keras.backend as K
import numpy as np
from keras import callbacks
from keras.engine.training import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers.core import Reshape, RepeatVector, Dropout
from keras.layers.merge import Dot, Concatenate, Add
from keras.layers.wrappers import TimeDistributed
from scipy.optimize._hungarian import linear_sum_assignment

import reader
from layer import SinkhornLayer

parser = argparse.ArgumentParser()
parser.add_argument("--predict-permutations", dest="predict_perms", type=bool, metavar='<bool>', default=True,
                    help="Whether to predict permutations directly ar to predict values [default=True]")
parser.add_argument("--temperature", dest="temp", type=float, metavar='<float>', default=1.0,
                    help="Sinkhorn layer temperature [default=1.0]")
parser.add_argument("--sinkhorn-iterations", dest="sinkhorn_iterations", type=int, metavar='<int>', default=20,
                    help="Sinkhorn iterations [default=20]")

parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=5,
                    help="Epochs number [default=5]")
parser.add_argument("--batch-size", dest="batch_size", type=int, metavar='<int>', default=32,
                    help="Batch size [default=32]")

parser.add_argument("--min-len", dest="min_len", type=int, metavar='<int>', default=10,
                    help="Min. length of the numbers sequence [default=5], the rest of the characters is padded")
parser.add_argument("--max-len", dest="max_len", type=int, metavar='<int>', default=10,
                    help="Max. length of the numbers sequence [default=10]")

parser.add_argument("--hidden-size", dest="hidden_dim", type=int, metavar='<int>', default=512,
                    help="Hidden layers dimensions [default=512]")

args = parser.parse_args()
min_len = args.min_len
max_len = args.max_len
PREDICT_VALUES = not args.predict_perms
HIDDEN_DIM = args.hidden_dim

reader.np
np.random.seed(100)

print('Loading data...')
(x_train, y_train, p_train), (x_test, y_test, p_test) = reader.generate_numbers(100000, 1000,
                                                                                min_length=min_len,
                                                                                sequence_max_length=max_len)

enc_p_train, enc_p_test = None, None

if not PREDICT_VALUES:
    print("Encoding permutations...")
    enc_p_train = reader.encode_perm(p_train)
    enc_p_test = reader.encode_perm(p_test)

print("Data shuffled so here we go")

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Building model...')

K.set_learning_phase(1)

# (?, length, input_vectors_size)
input = Input(shape=(max_len, 1))

# embedding the objects we are to sort
embedding = TimeDistributed(Dense(units=HIDDEN_DIM, kernel_regularizer="l2"))(input)
embedding = Dropout(rate=0.7)(embedding)

# pairwise scalar products to take object 'interactions' into account
dot = Dot([-1, -1])([embedding, embedding])

# reshaping into a single vector
interactions = Reshape(target_shape=(max_len * max_len,))(dot)

# two independent fully-connected layers with different activations
interactions1 = Dense(units=max_len * max_len, activation="sigmoid")(interactions)
interactions2 = Dense(units=max_len * max_len, activation="tanh")(interactions)

# (this trick seems to be an important one)
added_interactions = Add()([interactions1, interactions2])

# appending 'interactions' to embeddings
interactions_replicated = RepeatVector(max_len)(added_interactions)
joined = Concatenate(axis=-1)([embedding, interactions_replicated])

# dense layer for dense layer outputs of the size equal to length
layer_for_combining = TimeDistributed(Dense(units=max_len, activation="tanh", ),
                                      input_shape=(max_len, max_len ** 2 + max_len))(joined)

# permutation approximation layer
sinkhorn = SinkhornLayer(n_iters=args.sinkhorn_iterations, temperature=args.temp, name="sinkhorn")(layer_for_combining)

# recovery layer: PERM x X
permute_apply = Dot(axes=[-2, -2])([sinkhorn, input])

if PREDICT_VALUES:
    loss = "mse"
    metrics = ["mae"]
    labels_train = y_train
    labels_test = y_test
    resulting_layer = permute_apply
else:
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    labels_train = enc_p_train.astype(int)
    labels_test = enc_p_test.astype(int)
    resulting_layer = sinkhorn

model = Model(input, resulting_layer)
model.compile(loss=loss, optimizer="adam", metrics=metrics)

print(model.summary())

print("Fitting...")

history = model.fit(x_train, labels_train,
                    batch_size=args.batch_size, epochs=args.epochs,
                    verbose=1, validation_split=0.1,
                    callbacks=[callbacks.EarlyStopping(min_delta=0.00001, verbose=1),
                               callbacks.ReduceLROnPlateau(verbose=1)])

K.set_learning_phase(0)

print("********************************************************************")

N = 5

print(N, "prediction samples...")

get_layer_output = K.function([model.layers[0].input], [model.get_layer(name="sinkhorn").output])
np.set_printoptions(precision=3)

for (orig, pred), (true, perm) in zip(zip(x_test[:N], model.predict(x_test)[:N]), zip(y_test[:N], p_test)):

    layer_output = get_layer_output([[orig]])[0]
    assignment = linear_sum_assignment(-layer_output[0])

    print("---------------------------------------------------------------")
    print("True permutation:     \t", perm, end="\n")
    print("Predicted permutation:\t", assignment[1])
    print()

    np.set_printoptions(precision=3)

    if PREDICT_VALUES:
        print("True values:             \t", true.ravel())
    else:
        print("True values:             \t", true.ravel())

    print("Permuted with prediction:\t", orig.ravel()[reader.reverse_permutation(assignment[1])])

    print()
    print()

print("********************************************************************")

# checking the goodness of fit on train data
score_train = model.evaluate(x_train, labels_train, batch_size=args.batch_size, verbose=0)

print('Train LOSS:', model.loss, score_train[0])
print('Train METRIC:', model.metrics, score_train[1])
print()

# checking the quality on test data
score_test = model.evaluate(x_test, labels_test, batch_size=args.batch_size, verbose=0)

print('Test LOSS:', model.loss, score_test[0])
print('Test METRIC:', model.metrics, score_test[1])
