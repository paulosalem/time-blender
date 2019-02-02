import unittest

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
        self.common_model_test(BankingModels.salary_earner())

    def test_cycle(self):
        self.common_model_test(SimpleModels.cycle())

    def test_kondratiev_business_cycle(self):
        self.common_model_test(EconomicModels.kondratiev_business_cycle())

    def test_predator_prey(self):
        predators_model, preys_model = EcologyModels.predator_prey(n_predators=100, n_preys=100,
                                                                   alpha=1.01, beta=0.002, delta=2.2, gamma=0.002)
        self.common_model_test(predators_model)

