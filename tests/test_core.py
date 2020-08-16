import copy

import numpy as np

from tests.common import AbstractTest
from time_blender.coordination_events import Piecewise
from time_blender.core import Generator, ConstantEvent
from time_blender.deterministic_events import WalkEvent
from time_blender.models import BankingModels
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
