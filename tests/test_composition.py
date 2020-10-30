import unittest
import pandas as pd

from time_blender.coordination_events import OnceEvent, Choice
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

        rw = WalkEvent(e1, initial_pos=0.0)

        g = Generator(start_date=self.begin_date, end_date=self.end_date)

        data = g.generate({'rw': rw})
        print(data)
        self.assertEqual(len(data), self.days)

    def test_composition_2(self):
        norm = NormalEvent(0, 1)
        we = WaveEvent(10, 3)

        compos = norm + we

        data = Generator(start_date=self.begin_date, end_date=self.end_date).generate([compos])
        self.assertEqual(len(data), self.days)

    def test_composition_3(self):
        const = ConstantEvent(4)
        norm = NormalEvent(0, 1)
        we = WaveEvent(30, 3)
        t_change = OnceEvent(UniformEvent(0, 90))

        compos1 = we + norm
        compos2 = const + norm

        def aux(t, i, memory, sub_events):
            if i <= sub_events['t_change'].execute(t):
                return sub_events['compos1'].execute(t)
            else:
                return sub_events['compos2'].execute(t)

        e = LambdaEvent(aux, sub_events={'t_change': t_change, 'compos1': compos1, 'compos2': compos2})

        data = Generator(start_date=self.begin_date, end_date=self.end_date).generate([e])
        df = data
        self.assertEqual(len(df), self.days)

    def test_composition_4(self):
        const = ConstantEvent(4)
        we = WaveEvent(30, 3)

        compos1 = we
        compos2 = const

        chc = Choice([compos1, compos2])

        data = Generator(start_date=self.begin_date, end_date=self.end_date).generate([chc])
        self.assertEqual(len(data), self.days)
