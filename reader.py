# -*- coding: utf-8 -*-

import numpy as np
from keras.utils.np_utils import to_categorical

np.random.seed(1)


def generate_numbers(train, test, sequence_max_length=100, min_length=5):
    def gen(k):

        x, y, p = [], [], []

        for i in range(k):

            size = np.random.randint(min_length, sequence_max_length + 1)
            nums = np.sort(np.random.uniform(0, 1.0, size))

            if size < sequence_max_length:
                nums = np.hstack([- np.ones(sequence_max_length - nums.shape[0]), nums])

            perm = np.array(np.random.permutation(range(sequence_max_length)))
            perm_nums = nums[perm]

            x.append(np.matrix(perm_nums).T)
            y.append(np.matrix(nums).T)
            p.append(perm)

        return np.array(x), np.array(y), np.array(p)

    train_data = gen(train)
    test_data = gen(test)

    return train_data, test_data


def reverse_permutation(p):
    perm_reverse = np.zeros_like(p)

    for i in range(p.shape[0]):
        perm_reverse[p[i]] = i

    return perm_reverse


def encode_perm(data):

    enc = []

    for row in data:
        enc.append(to_categorical(row))

    return np.asarray(enc).astype(int)
