import unittest
import pandas as pd
import numpy as np
import math
import random

from time_blender.core import Generator


class AbstractTest(unittest.TestCase):

    def common_model_test(self, model, n=1, print_data=False):
        data = Generator(start_date=self.start_date, end_date=self.end_date, freq='D').generate([model], n=n)

        idx_true = pd.date_range(self.start_date, self.end_date, freq='D')

        if n > 1:
            df = data[0]
        else:
            df = data

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
            print(f"Is {a} close to {b} up to {rel_tol} relative tolerance and {abs_tol} absolute tolerance?")

        self.assertTrue(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol))

    def generate_learn_generate(self, generator, original_event, fresh_event,
                                start_date, end_date,
                                n_simulations=3, max_optimization_evals=300,
                                upper_bound=None, lower_bound=None,
                                error_strategy='best_trace'):

        data = generator.generate(events=original_event)
        values = data.values

        fresh_event.generalize_from_observations([values], n_simulations=n_simulations,
                                                 max_optimization_evals=max_optimization_evals,
                                                 upper_bound=upper_bound, lower_bound=lower_bound,
                                                 error_strategy=error_strategy,
                                                 verbose=True)

        return Generator(start_date=start_date, end_date=end_date).generate(fresh_event).values