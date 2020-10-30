import unittest
import pandas as pd
import numpy as np
import math

from tests.common import AbstractTest
from time_blender.coordination_events import Piecewise, Choice, SeasonalEvent, OnceEvent, PastEvent, \
    ConcludingValueEvent
from time_blender.core import Generator, ConstantEvent, LambdaEvent
from time_blender.deterministic_events import WalkEvent
from time_blender.random_events import NormalEvent, UniformEvent, PoissonEvent


class TestOnceEvent(AbstractTest):
    def test(self):
        once = OnceEvent(NormalEvent(0, 1))
        data = self.common_model_test(once, print_data=True)

        # All values are the same, because the first value was recorded by Once and just repeated later
        self.assertEquals(set(list(data.diff().dropna().values[:, 0])), {0.0})


class TestPastEvent(AbstractTest):
    def test(self):
        walk = WalkEvent(0, 10)
        past_walk = PastEvent(3, refers_to=walk, parallel_events=[walk])

        data_past_walk = self.common_model_test(past_walk)
        print(walk._generated_values)
        walk.reset()
        data_walk = self.common_model_test(walk)

        # The present is greater than the past
        #print(data_walk[0])
        #print(data_past_walk[0])
        #self.assertTrue((data_walk[0] > data_past_walk[0]).all().values[0])
        self.assertTrue((data_walk.iloc[:, 0] > data_past_walk.iloc[:, 0]).all())

        # The difference is 30, since the walk steps 10 and the past event is 3 steps behind.
        self.assertEquals(set((data_walk.iloc[10:, 0] - data_past_walk.iloc[10:, 0]).values), {30.0})

class TestConcludingValueEvent(AbstractTest):

    def test(self):
        event = ConcludingValueEvent() + ConstantEvent(10)
        self.common_model_test(event, n=1, print_data=True)

class TestSeasonalEvent(AbstractTest):
    def test(self):
        data_1 = self.common_model_test(SeasonalEvent(NormalEvent(0, 1), base=0, year=2018, month=9,
                                             day=None, hour=None, minute=None, second=None,
                                             name=None, parallel_events=None), print_data=False)

        data_2 = self.common_model_test(SeasonalEvent(NormalEvent(0, 1), base=0, year=None, month=None,
                                                      day=None, hour=None, minute=None, second=None,
                                                      is_weekday=True,
                                                      name=None, parallel_events=None), print_data=False)

        data_3 = self.common_model_test(SeasonalEvent(NormalEvent(0, 1), base=None, year=None, month=None,
                                                      day=None, hour=None, minute=None, second=None,
                                                      is_weekday=True,
                                                      name=None, parallel_events=None), print_data=False)

        print(data_2)
        print(data_3)



class TestChoice(AbstractTest):
    def test(self):
        self.common_model_test(Choice([NormalEvent(0, 1), NormalEvent(0, 4)]))


class TestPiecewise(AbstractTest):
    def test(self):
        self.common_model_test(Piecewise([NormalEvent(100, 10), NormalEvent(0, 1)],
                                         t_separators=[NormalEvent(30, 20)]),
                               n=10)
