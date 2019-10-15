# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



def generate_simulated_data(num_instances=200, num_labels=2, num_classifiers=71):
    if num_labels <= 1:
        y = np.random.randint(2, size=num_instances) * 2 - 1
        yt = np.random.randint(2, size=[num_classifiers, num_instances]) * 2 - 1
        return y.tolist(), yt.tolist()
    y = np.random.randint(num_labels, size=num_instances)
    yt = np.random.randint(num_labels, size=[num_classifiers, num_instances])
    return y.tolist(), yt.tolist()


def negative_generate_simulate(num_instances, num_classifiers):
    y1, yt1 = generate_simulated_data(num_instances, 2, num_classifiers)
    y2 = (np.array(y1) * 2 - 1).tolist()
    yt2 = (np.array(yt1) * 2 - 1).tolist()
    return y1, yt1, y2, yt2


