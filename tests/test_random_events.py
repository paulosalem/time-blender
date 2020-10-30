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
        super().setUp()
        self.param_1 = -20
        self.param_2 = 20
        self.original_event = UniformEvent(low=self.param_1, high=self.param_2)
        self.generator = Generator(start_date=self.start_date, end_date=self.end_date)

    def test_execute(self):
        data = self.generator.generate(self.original_event)
        values = data.values
        mean = np.mean(values)
        self.assertClose(mean, 0, abs_tol=1.0)

        self.common_model_test(self.original_event)

    def test_generalize_from_observations(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  original_event=self.original_event,
                                                  fresh_event=UniformEvent(low=NormalEvent(-25, 10),
                                                                           high=NormalEvent(2, 5)),
                                                  max_optimization_evals=300,
                                                  start_date=self.start_date, end_date=self.end_date)

        print(fresh_data)
        self.assertClose(np.mean(fresh_data), 0.0, rel_tol=0.1, abs_tol=1.0)

    # TODO test_generalize_from_observations_2 using indexes
    def test_generalize_from_observations_2(self):
        data = self.generator.generate(events=self.original_event)
        fresh_event = UniformEvent(low=NormalEvent(-25, 10),
                                   high=NormalEvent(2, 5))

        # data contains Pandas Series
        fresh_event.generalize_from_observations(data, n_simulations=3, max_optimization_evals=300,
                                                 error_strategy='best_trace')

        fresh_data = Generator(start_date=self.start_date, end_date=self.end_date).generate(fresh_event)

        print(data)
        print(fresh_data)

        # TODO test index intersections when observations are Pandas' Series
        self.assertTrue(False) # TODO


class TestNormalEvent(AbstractRandomEventTest):

    def setUp(self):
        super().setUp()
        self.param_1 = 14
        self.original_event = NormalEvent(self.param_1, 1.0)
        self.generator = Generator(start_date=self.start_date, end_date=self.end_date)

    def test_execute(self):
        data = self.generator.generate(self.original_event)
        values = data.values
        mean = np.mean(values)
        print(mean)
        self.assertClose(mean, self.param_1, rel_tol=0.1)

        self.common_model_test(self.original_event)

    def test_generalize_from_observations_1(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  original_event=self.original_event,
                                                  fresh_event=NormalEvent(10.0, 2.0),
                                                  start_date=self.start_date, end_date=self.end_date,
                                                  max_optimization_evals=300,
                                                  error_strategy='all_traces')

        self.assertClose(np.mean(fresh_data), self.param_1, rel_tol=0.1)

    def test_generalize_from_observations_2(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  original_event=self.original_event,
                                                  fresh_event=NormalEvent(UniformEvent(low=-100.0, high=100.0), 2.0),
                                                  n_simulations=3, max_optimization_evals=500,
                                                  start_date=self.start_date, end_date=self.end_date)

        self.assertClose(np.mean(fresh_data), self.param_1, rel_tol=0.1)


class TestPoissonEvent(AbstractRandomEventTest):

    def setUp(self):
        super().setUp()
        self.param_1 = 7
        self.original_event = PoissonEvent(self.param_1)
        self.generator = Generator(start_date=self.start_date, end_date=self.end_date)

    def test_execute(self):
        data = self.generator.generate(self.original_event)
        values = data.values
        mean = np.mean(values)
        print(mean)

        self.assertClose(mean, self.param_1)

        self.common_model_test(self.original_event)

    def test_generalize_from_observations(self):

        fresh_data = self.generate_learn_generate(self.generator,
                                                  original_event=self.original_event,
                                                  fresh_event=PoissonEvent(UniformEvent(low=1.0,
                                                                                        high=10.0)),
                                                  start_date=self.start_date, end_date=self.end_date)

        self.assertClose(np.mean(fresh_data), self.param_1)


class TestTopResistance(AbstractRandomEventTest):

    def test_execute(self):
        base = WalkEvent(NormalEvent(1, 2))
        resistance_1 = NormalEvent(0.5, 0.1)
        model = TopResistance(base,
                              resistance_value_begin=50,
                              resistance_value_end=55,
                              resistance_probability=0.5,
                              resistance_strength_event=resistance_1)

        self.common_model_test(model)


class TestBottomResistance(AbstractRandomEventTest):

    def test_execute(self):
        base = WalkEvent(NormalEvent(-1, 2))
        resistance_1 = NormalEvent(0.5, 0.1)
        model = BottomResistance(base,
                                 resistance_value_begin=-20,
                                 resistance_value_end=-30,
                                 resistance_probability=0.5,
                                 resistance_strength_event=resistance_1)

        self.common_model_test(model)

    def test_execute_2(self):
        resistance_1 = NormalEvent(0.5, 0.1)
        base = NormalEvent(19, 5)
        model = TopResistance(base,
                      resistance_value_begin=20,
                      resistance_value_end=50,
                      resistance_probability=0.9,
                      resistance_strength_event=resistance_1)

        data = self.common_model_test(model)
        print(data)