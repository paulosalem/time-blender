import numpy as np

from tests.common import AbstractTest
from time_blender.core import Generator, ConstantEvent
from time_blender.deterministic_events import WalkEvent
from time_blender.random_events import NormalEvent, UniformEvent, PoissonEvent, TopResistance, BottomResistance


class AbstractRandomEventTest(AbstractTest):

    def setUp(self):
        super().setUp()


class TestUniformEvent(AbstractRandomEventTest):

    def setUp(self):
        self.param_1 = -20
        self.param_2 = 20
        self.event = UniformEvent(low=self.param_1, high=self.param_2)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        mean = np.mean(values)
        self.assertClose(mean, 0, abs_tol=1.0)

        self.common_model_test(self.event)

    def test_generalize_from_observations(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  UniformEvent(low=NormalEvent(-25, 10), high=NormalEvent(25, 10)),
                                                  self.start_date, self.end_date)

        print(fresh_data)
        self.assertClose(np.mean(fresh_data), 0.0, abs_tol=2.0)


class TestNormalEvent(AbstractRandomEventTest):

    def setUp(self):
        self.param_1 = 14
        self.event = NormalEvent(self.param_1, 1.0)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        mean = np.mean(values)
        print(mean)
        self.assertClose(mean, self.param_1)

        self.common_model_test(self.event)

    def test_generalize_from_observations_1(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  NormalEvent(10.0, 2.0),
                                                  self.start_date, self.end_date)

        self.assertClose(np.mean(fresh_data), self.param_1)

    def test_generalize_from_observations_2(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  NormalEvent(UniformEvent(low=-100.0, high=100.0), 2.0),
                                                  self.start_date, self.end_date)

        self.assertClose(np.mean(fresh_data), self.param_1)


class TestPoissonEvent(AbstractRandomEventTest):

    def setUp(self):
        self.param_1 = 7
        self.event = PoissonEvent(self.param_1)
        self.generator = Generator(self.event)
        super().setUp()

    def test_execute(self):
        data = self.generator.generate(self.start_date, self.end_date)
        values = data[0].values
        mean = np.mean(values)
        print(mean)

        self.assertClose(mean, self.param_1)

        self.common_model_test(self.event)

    def test_generalize_from_observations(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  PoissonEvent(UniformEvent(low=ConstantEvent(0.0,
                                                                                              require_lower_bound=0.0),
                                                                            high=10.0)),
                                                  self.start_date, self.end_date)

        self.assertClose(np.mean(fresh_data), self.param_1)


class TestTopResistance(AbstractRandomEventTest):

    def test_execute(self):
        base = WalkEvent(0, NormalEvent(1, 2))
        resistance_1 = NormalEvent(0.5, 0.1)
        model = TopResistance(base,
                              resistance_value_begin=50,
                              resistance_value_end=55,
                              resistance_probability=0.5,
                              resistance_strength_event=resistance_1)

        self.common_model_test(model)


class TestBottomResistance(AbstractRandomEventTest):

    def test_execute(self):
        base = WalkEvent(0, NormalEvent(-1, 2))
        resistance_1 = NormalEvent(0.5, 0.1)
        model = BottomResistance(base,
                                 resistance_value_begin=-20,
                                 resistance_value_end=-30,
                                 resistance_probability=0.5,
                                 resistance_strength_event=resistance_1)

        self.common_model_test(model)