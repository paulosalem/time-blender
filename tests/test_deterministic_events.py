import unittest
import pandas as pd
import numpy as np

from tests.common import AbstractTest
from time_blender.core import Generator, ConstantEvent
from time_blender.deterministic_events import ClockEvent, WaveEvent, WalkEvent


class TestClockEvent(AbstractTest):

    def setUp(self):
        self.event = ClockEvent()
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values

        # values must be increasing
        for i in range(1, len(values)):
            self.assertEquals(values[i], values[i - 1] + 1)


class TestWaveEvent(AbstractTest):

    def setUp(self):
        self.event = WaveEvent(30, 100)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        print(np.mean(values))
        self.assertClose(np.mean(values), 0.0, abs_tol=2.0) # centers on zero
        self.assertGreater(len([v for v in values if v > 90]), 0) # goes up
        self.assertGreater(len([v for v in values if v < 90]), 0) # goes down


class TestConstantEvent(AbstractTest):

    def setUp(self):
        self.event = ConstantEvent(30)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        for v in values:
            self.assertEquals(v, 30)


class TestWalkEvent(AbstractTest):

    def setUp(self):
        self.event = WalkEvent(0, 10)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        print(values)