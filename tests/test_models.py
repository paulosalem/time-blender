import unittest
import math
import pandas as pd
pd.set_option('display.max_rows', None)

from tests.common import AbstractTest
from time_blender.models import ClassicModels, BankingModels, SimpleModels, EconomicModels, EcologyModels

class ModelsTest(AbstractTest):

    def test_ar(self):
        self.common_model_test(ClassicModels.ar(3, random_seed=42))
        self.common_model_test(ClassicModels.ar(3, coefs=[0, 1, 2]))

    def test_ma(self):
        self.common_model_test(ClassicModels.ma(3))

    def test_arma(self):
        self.common_model_test(ClassicModels.arma(p=3, q=3))

    def test_salary_earner(self):

        s = self.common_model_test(BankingModels.salary_earner(salary=5000, payment_day=1))[0].iloc[:, 0]
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

