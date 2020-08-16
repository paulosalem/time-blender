import random

import pandas as pd
import numpy as np

# A counter to be used when creating fresh names
import time_blender


def fresh_id():
    fresh_id.counter += 1
    return fresh_id.counter

fresh_id.counter = -1


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # Add any other source of randomness here


def shift_weekend_and_holidays(day, direction='forward', holidays=[]):
    shift = 0
    if (day.weekday() == 5) or (day.weekday() == 6) or (day in holidays):
        if direction == 'forward':
            shift = 1
        else:
            shift = -1

        # recursive step
        return shift_weekend_and_holidays(day + pd.DateOffset(days=shift), direction, holidays)

    else:
        # recursion base
        return day

