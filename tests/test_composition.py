import unittest
import pandas as pd

from time_blender.coordination_events import Once, Choice
from time_blender.core import Generator, LambdaEvent, ConstantEvent
from time_blender.deterministic_events import WalkEvent, WaveEvent
from time_blender.random_events import NormalEvent, UniformEvent


class TestDeterministicEvents(unittest.TestCase):
    def setUp(self):
        self.begin_date = pd.Timestamp(2018, 1, 1)
        self.end_date = pd.Timestamp(2018, 3, 30)
        self.days = (self.end_date - self.begin_date).days + 1

    def test_composition_1(self):
        e1 = NormalEvent(0.0, 2)

        rw = WalkEvent(0.0, e1)

        g = Generator({'rw': rw})

        data = g.generate(self.begin_date, self.end_date, 1)
        print(data[0])
        self.assertEqual(len(data[0]), self.days)

    def test_composition_2(self):
        norm = NormalEvent(0, 1)
        we = WaveEvent(10, 3)

        compos = norm + we

        data = Generator([compos]).generate(self.begin_date, self.end_date, 1)
        print(data[0])
        self.assertEqual(len(data[0]), self.days)

    def test_composition_3(self):
        const = ConstantEvent(4)
        norm = NormalEvent(0, 1)
        we = WaveEvent(30, 3)
        t_change = Once(UniformEvent(0, 90))

        compos1 = we + norm
        compos2 = const + norm

        def aux(t, i, memory):
            if i <= t_change.execute(t):
                return compos1.execute(t)
            else:
                return compos2.execute(t)

        e = LambdaEvent(aux, sub_events=[compos1, compos2])

        data = Generator([e]).generate(self.begin_date, self.end_date, 1)
        df = data[0]
        print(df)
        self.assertEqual(len(df), self.days)

    def test_composition_4(self):
        const = ConstantEvent(4)
        we = WaveEvent(30, 3)

        compos1 = we
        compos2 = const

        chc = Choice([compos1, compos2])

        data = Generator([chc]).generate(self.begin_date, self.end_date, 1)
        print(data[0])
        self.assertEqual(len(data[0]), self.days)
