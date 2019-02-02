import random

import pymc3 as pm
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


class PyMC3Utils:

    @staticmethod
    def multi_switch( vars, categorical_var):
        """
        Selects a variable in the specified set according to the categorical distribution provided.
        :param vars: a collection of PyMC3 random variables or scalars.
        :param categorical_var: a PyMC3 categorical variable.
        :return: a composite PyMC3 switch random variable.
        """
        vars = set(vars)

        def aux(vars, i=0):
            if len(vars) > 2:
                var = vars.pop()
                return pm.math.switch(categorical_var == i,
                                      var,
                                      aux(vars, i + 1))
            else:
                return vars.pop()

        return aux(vars)


