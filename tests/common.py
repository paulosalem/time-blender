import unittest
import pandas as pd
import numpy as np
import math
import random

from time_blender.core import Generator


class AbstractTest(unittest.TestCase):

    def common_model_test(self, model, print_data=False):
        data = Generator([model]).generate(self.start_date, self.end_date, 1, freq='D')

        idx_true = pd.date_range(self.start_date, self.end_date, freq='D')

        df = data[0]
        self.assertEqual(len(df), len(idx_true))
        if print_data:
            print(df)

        return data

    def setUp(self):
        self.start_date = '2016-01-01'
        self.end_date = '2018-10-01'

        # set random seeds for consistent results
        np.random.seed(3)
        random.seed(3)

    def assertClose(self, a, b, rel_tol=0.1, abs_tol=0.0, verbose=True):
        if verbose:
            print(f"Is {a} close to {b} up to {rel_tol} relative tolerance or {abs_tol} absolute tolerance?")

        self.assertTrue(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol))

    def generate_learn_generate(self, generator, fresh_event, start_date, end_date):
        data = generator.generate(start_date, end_date)
        values = data[0].values

        fresh_event.generalize_from_observations([values], max_optimization_evals=60)
        return Generator(fresh_event).generate(start_date, end_date)[0].values