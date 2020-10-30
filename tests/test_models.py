import unittest
import math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from time_blender.core import ConstantEvent, Generator
from time_blender.random_events import NormalEvent

pd.set_option('display.max_rows', None)

from tests.common import AbstractTest
from time_blender.models import ClassicModels, BankingModels, SimpleModels, EconomicModels, EcologyModels

class ModelsTest(AbstractTest):

    def setUp(self):
        super().setUp()
        self.start_date = '2016-01-01'
        self.end_date = '2016-03-01'

    def test_ar(self):
        self.common_model_test(ClassicModels.ar(3, random_seed=42))
        self.common_model_test(ClassicModels.ar(3, coefs=[0, 1, 2]))

    def test_ma(self):
        self.common_model_test(ClassicModels.ma(3))

    def test_arma(self):
        self.common_model_test(ClassicModels.arma(p=3, q=3))

    def test_salary_earner(self):

        s = self.common_model_test(BankingModels.salary_earner(salary_value=20000, payment_day=1)).iloc[:, 0]
        s = s.diff()
        s.iloc[0] = 20000.0  # we must add an extra salary because the diff operation removed the first one

        n_months = len(s.resample('MS').sum())
        n_salaries = len(s[s >= 15000])  # this assumes that daily expense in payment day is not very large
        print(n_months, n_salaries)

        # the number of salaries payed must be equal to the number of months considered
        self.assertEqual(n_months, n_salaries)

    def test_salary_earner_simple(self):

        s = self.common_model_test(BankingModels.salary_earner_simple(salary_value=5000, payment_day=1)).iloc[:, 0]
        s = s.diff()
        s.iloc[0] = 5000.0  # we must add an extra salary because the diff operation removed the first one

        n_months = len(s.resample('MS').sum())
        n_salaries = len(s[s == 5000])
        print(n_months, n_salaries)

        # the number of salaries payed must be equal to the number of months considered
        self.assertEqual(n_months, n_salaries)

    def test_cycle(self):
        self.common_model_test(SimpleModels.cycle())

    def test_kondratiev_business_cycle(self):
        self.common_model_test(EconomicModels.kondratiev_business_cycle())

    def test_predator_prey(self):
        predators_model, preys_model = EcologyModels.predator_prey(n_predators=100, n_preys=100,
                                                                   alpha=1.01, beta=0.002, delta=2.2, gamma=0.002)
        self.common_model_test(predators_model)

    def test_generalize_from_observations_1(self):
        oracle = ClassicModels.arima(2, 0, constant=ConstantEvent(5))
        oracle_data = self.common_model_test(oracle)

        model_under_test = ClassicModels.arima(2, 0, constant=ConstantEvent(1))
        generator = Generator(start_date=self.start_date, end_date=self.end_date, freq='D')
        fresh_data = self.generate_learn_generate(generator,
                                                  original_event=oracle,
                                                  fresh_event=model_under_test,
                                                  start_date=self.start_date, end_date=self.end_date,
                                                  n_simulations=1, max_optimization_evals=1000,
                                                  error_strategy='best_trace')

        print(np.mean(fresh_data), np.mean(oracle_data))
        print(mean_absolute_error(fresh_data, oracle_data))
        self.assertClose(np.mean(fresh_data), np.mean(oracle_data), rel_tol=0.2)