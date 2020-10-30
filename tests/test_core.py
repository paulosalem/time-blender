import copy

import numpy as np

from tests.common import AbstractTest
from time_blender.coordination_events import Piecewise, Replicated, PastEvent
from time_blender.core import Generator, ConstantEvent, LambdaEvent
from time_blender.deterministic_events import WalkEvent
from time_blender.models import BankingModels, ClassicModels
from time_blender.random_events import NormalEvent, UniformEvent, PoissonEvent, TopResistance, BottomResistance


class TestEvent(AbstractTest):

    def setUp(self):
        super().setUp()

    def test_clone(self):
        #
        # Let's test the copy strategy using a piecewise model.
        #
        banking_model_1 = BankingModels.salary_earner(salary=ConstantEvent(5000.0,
                                                                           require_lower_bound=0,
                                                                           require_upper_bound=30000),
                                                      expense_mean=ConstantEvent(100.0,
                                                                                 require_lower_bound=0,
                                                                                 require_upper_bound=1000),
                                                      expense_sd=ConstantEvent(100.0,
                                                                               require_lower_bound=0,
                                                                               require_upper_bound=30000))

        banking_model_2 = banking_model_1.clone()
        banking_model_3 = banking_model_1.clone()

        # top level names must be unique
        self.assertNotEqual(banking_model_1.name, banking_model_2.name)
        self.assertNotEqual(banking_model_1.name, banking_model_3.name)
        self.assertNotEqual(banking_model_2.name, banking_model_3.name)

        # nested names must also be unique
        self.assertNotEqual(banking_model_1._causal_parameters[0].name,
                            banking_model_2._causal_parameters[0].name)

        # classes must be equal, though
        self.assertEqual(banking_model_1._causal_parameters[0].__class__,
                         banking_model_2._causal_parameters[0].__class__)

        t_separator_1 = NormalEvent(ConstantEvent(60.0,
                                                  require_lower_bound=0,
                                                  require_upper_bound=100),
                                    ConstantEvent(20.0,
                                                  require_lower_bound=0,
                                                  require_upper_bound=100))

        t_separator_2 = t_separator_1.clone()

        # top level names must be unique
        self.assertNotEqual(t_separator_1.name, t_separator_2.name)

        pw = Piecewise([banking_model_1, banking_model_2, banking_model_3],
                       t_separators=[t_separator_1, t_separator_2])

        res = self.common_model_test(pw)

    def test_clone_2(self):
        base_event = NormalEvent() + PoissonEvent()

        def aux(t, i, memory, sub_events):
            res = 2 * sub_events['base'].execute(t)
            return res

        print(aux.__closure__)

        base_model = LambdaEvent(aux, sub_events={'base': base_event})

        model = Replicated(base_model, NormalEvent(mean=10, std=5), max_replication=2)

        data = self.common_model_test(model, n=2)

        self.assertTrue((data[0].iloc[-10:-1].values != data[1].iloc[-10:-1].values).any())

    def test_clone_3(self):
        pe = PastEvent(1)
        event = ConstantEvent(1) + pe
        pe.refers_to(event)

        self.common_model_test(event)

        # cloning must not break anything
        cloned_event = event.clone()
        self.common_model_test(cloned_event)

        self.assertNotEqual(event._causal_parameters[0].name, cloned_event._causal_parameters[0].name)

    def test_constant_generation(self):
        constant_event_1 = ConstantEvent(10)
        data_1 = self.common_model_test(constant_event_1, n=2)

        # series generated from a constant must have the same values
        self.assertTrue((data_1[0].iloc[-10:-1].values == data_1[1].iloc[-10:-1].values).all())

    def test_lambda_composition_generation(self):
        #
        # Various composition strategies
        #
        events = [NormalEvent(0, 1),
                  NormalEvent(0, 1)*ConstantEvent(1000),
                  NormalEvent(0, 1)+ConstantEvent(1000),
                  NormalEvent(0, 1)-ConstantEvent(1000),
                  NormalEvent(0, 1)/ConstantEvent(1000)]

        data_sets = [self.common_model_test(e, n=2) for e in events]

        # check each composition behavior
        for data in data_sets:
            # different generated data series must have different values
            #print(data[0].iloc[-10:-1].values, data[1].iloc[-10:-1].values)
            self.assertFalse((data[0].iloc[-10:-1].values == data[1].iloc[-10:-1].values).all())
