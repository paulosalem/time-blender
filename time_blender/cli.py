########################################################################################################################
# Command-Line Interface utils
########################################################################################################################
import pandas as pd
from sigtools.wrappers import decorator, wrapper_decorator
from clize import run, parser, Parameter
import ast
import matplotlib.pyplot as plt

import time_blender
from time_blender.util import set_random_seed


@parser.value_converter
def a_list(x):
    if isinstance(x, list):
        return x
    else:
        return ast.literal_eval(x)


using_cli = False
cli_end_date = pd.Timestamp.now()
cli_begin_date = cli_end_date - pd.DateOffset(months=12)
cli_output_file = None #'out.csv'
cli_freq = 'D'
cli_plot = False


@wrapper_decorator
def cli_model(model_func, *args, n: int=1, begin=None, end=None, freq='D', output_file=None, plot=False,
              random_seed=None, **kwargs: Parameter.IGNORE):
    """

    :param model_func:
    :param args:
    :param n: How many series are to be generated with the specified parameters.
    :param begin: The first date to be generated. Format: YYYY-MM-DD
    :param end: The last date to be generated. Format: YYYY-MM-DD
    :param freq: The frequency of the generation. Refer to Pandas documentation for the proper string values.
    :param output_file: A prefix to be used as the base of files in which to save the series. If this is omitted,
                        the series is just returned in the console.
    :param plot: Whether the series should be plotted. If an output file is defined, the base name is also used
                 to save an image.
    :param kwargs:
    :return:
    """
    if using_cli:

        set_random_seed(random_seed)

        event = model_func(*args, **kwargs)

        if end is None:
            cli_end_date = pd.Timestamp.now()
        else:
            cli_end_date = pd.to_datetime(end)

        if begin is None:
            cli_begin_date = cli_end_date - pd.DateOffset(months=12)
        else:
            cli_begin_date = pd.to_datetime(begin)

        cli_freq = freq
        cli_output_file = output_file

        g = time_blender.core.Generator([event])
        print(cli_end_date, cli_end_date)
        data = g.generate(cli_begin_date, cli_end_date, n=n, freq=cli_freq)
        #print(data)
        results = None
        for i, d in enumerate(data):
            if cli_output_file is not None:
                if i == 0:
                    name = f'{cli_output_file}'
                else:
                    name = f'{i}_{cli_output_file}'

                d.to_csv(name + '.csv')
                plt.ioff()
                plt.clf()
                d.plot()
                plt.savefig(name + '.png')

            else:
                if results is None:
                    results = ""
                results += d.to_csv() # without a path, returns a string with the CSV

                if plot:
                    d.plot()


        # Shows all plots at once
        if cli_output_file is None:
            plt.show()

        return results

    else:
        return model_func(*args, **kwargs)

