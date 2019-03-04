import numpy as np


def assert_list_items(list_1, list_2):
    check_list = [True if np.isnan(list_1[i]) and np.isnan(list_2[i]) else list_1[i] == list_2[i]
                  for i in range(len(list_1))]
    assert np.all(check_list)