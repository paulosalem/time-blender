import unittest

import pandas as pd

from time_blender.util import shift_weekend_and_holidays


class TestUtil(unittest.TestCase):

    def test_shift_weekend_and_holidays(self):

        def aux_check(a, b, direction='forward'):
            shifted_day = shift_weekend_and_holidays(a, direction=direction, holidays=[])
            self.assertEquals(shifted_day, b)

        aux_check(pd.Timestamp(2018, 12, 22), pd.Timestamp(2018, 12, 24))
        aux_check(pd.Timestamp(2018, 12, 23), pd.Timestamp(2018, 12, 24))
        aux_check(pd.Timestamp(2018, 12, 24), pd.Timestamp(2018, 12, 24))

        aux_check(pd.Timestamp(2018, 12, 22), pd.Timestamp(2018, 12, 21), direction='backward')
        aux_check(pd.Timestamp(2018, 12, 23), pd.Timestamp(2018, 12, 21), direction='backward')