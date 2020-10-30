import logging
import math
import random
import copy

import numpy as np
from numpy.random import random_integers
import pandas as pd
import functools

import matplotlib
import matplotlib.pyplot as plt

from hyperopt import STATUS_OK, STATUS_FAIL
from scipy import signal
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import hyperopt
from joblib import Parallel, delayed

from time_blender.config import LEARNING_CONFIG
from time_blender.util import fresh_id, is_sequence


###############################################################################
# Auxiliary functions.
###############################################################################
def wrapped_constant_param(prefix, name, value, **kwargs):
    """
    Wraps the specified value as a ConstantEvent if it is numeric and not already an event.

    :param prefix: A prefix for the name of the event.
    :param name: The name of the event.
    :param value: The value to be wrapped.
    :param kwargs: Additional aruments to pass to ConstantEvent's constructor.
    :return: The wrapped value.
    """

    # default result
    res = value

    if not isinstance(value, Event):
        if isinstance(value, int) or isinstance(value, float):
            res = ConstantEvent(value, name=f'{prefix}_{name}', **kwargs)
        elif isinstance(value, bool):
            res = ConstantEvent(int(value), name=f'{prefix}_{name}', **kwargs)
        elif is_sequence(value):
            res = [wrapped_constant_param(prefix, f'{name}_{i}', v, **kwargs) for i, v in enumerate(value)]

    return res


###############################################################################
# Auxiliary structures
###############################################################################
class Invariant:

    def __init__(self, invariant_func, events, description=""):
        self._invariant_func = invariant_func
        self.description = description

        if isinstance(events, dict):
            self._named_events = events
        else:
            self._named_events = {}
            for event in events:
                assert event.name not in self._named_events, f"Event names cannot be duplicated, but we found " \
                                                             f"two named '{event.name}'"
                self._named_events[event.name] = event

    def check(self, t) -> (bool, str):
        return self._invariant_func(t, self._named_events), self.description


###############################################################################
# Core event classes
###############################################################################
class Event:

    # TODO remove push_down, since it must always be True?
    def __init__(self, name=None, parallel_events=None, push_down=True, allow_learning=True, invariants=None):
        self.name = self._default_name_if_none(name)

        self._indexed_generated_values = {}  # time to value
        self._generated_values = []  # sequence of values
        self._last_pos = -1
        self._allow_learning = allow_learning
        self._invariants = invariants

        # determine which attributes are causal events
        self._init_causal_parameters()

        # set parallel events
        self.parallel_events = None
        if parallel_events is not None:
            self.parallel_to(parallel_events)
        else:
            self.parallel_events = None

        self.push_down = push_down
        self.parent_value = None

        # to avoid infinite recursions, locks are available for some methods
        self._execution_locked = False
        self._push_down_locked = False

        # scaling parameters, used to rescale the generated values if needed
        self._scaling_max = None
        self._scaling_min = None


    def _init_causal_parameters(self):
        self._causal_parameters = []
        for k, v in self.__dict__.items():
            # events might be given as attributes
            if isinstance(v, Event):
                self._causal_parameters.append(v)

            # events might be stored in lists
            elif isinstance(v, list) and k != '_causal_parameters':  # the key verification avoids an infinite loop
                for element in v:
                    if isinstance(element, Event):
                        self._causal_parameters.append(element)

    def _wrapped_param(self, prefix, name, value, **kwargs):
        return wrapped_constant_param(prefix=prefix, name=name, value=value, **kwargs)

    def _causal_parameters_closure(self, only_learnable=True):
        closure = []
        for event in self._causal_parameters:
            # should the event be added?
            if only_learnable and (not event._allow_learning):
                add = False
            else:
                add = True

            # if so, add it
            if add:
                closure.append(event)
                closure = closure + event._causal_parameters_closure(only_learnable=only_learnable)

        return closure

    def execute(self, t):
        """
        Executes the event and generates an output for the present moment.

        :param t: The time in which the event takes place.
        :return: The scalar value of the event in the specified moment.
        """

        ######################
        # Auxiliary functions
        ######################

        def aux_check_invariants(t):
            invariants_hold, failed_invariant_description = self._check_invariants(t)
            if not invariants_hold:
                raise AssertionError(f"Invariant violated: {failed_invariant_description}")

        ###################
        # Main execute
        ###################
        if not self._execution_locked:
            self._execution_locked = True

            # check whether invariants hold
            aux_check_invariants(t)

            # update parallel events
            if self.parallel_events is not None:
                for e in self.parallel_events:
                    e.execute(t)

            # process this event
            if t not in self._indexed_generated_values:
                self._last_pos += 1

                v = self._execute(t, self._last_pos)

                # save the value for future reference and to avoid recomputing
                self._indexed_generated_values[t] = v
                self._generated_values.append(v)

            res = self._indexed_generated_values[t]

            # Result might be used by underlying events as well. So it must be pushed down.
            self._push_down(t, res)

            self._execution_locked = False

            return res

        else:
            return None

    def _execute(self, t, i):
        raise NotImplementedError("Must be implemented by concrete subclasses.")

    def add_invariant(self, invariant: Invariant):
        if self._invariants is None:
            self._invariants = []

        self._invariants.append(invariant)

        return self

    def add_invariants(self, invariants):
        for invariant in invariants:
            self.add_invariant(invariant)

    def _check_invariants(self, t):
        if self._invariants is not None:
            for invariant in self._invariants:
                holds, description = invariant.check(t)
                if not holds:
                    return False, description  # at least one invariant failed

            return True, ""  # no invariant failed, so they all hold

        else:
            return True, ""  # nothing to possibly fail

    def _push_down(self, t, parent_value):
        """
        Given a value executed at the specified moment by a parent event, pushes it down to its children
        events. That is to say, provides a downward path for executed values, opposite to the the regular
        information flow from children to parent.

        :param t: The execution moment.
        :param parent_value: The value executed by a parent.
        :return: None
        """

        # to avoid infinite loops in situations of mutual dependency, reentrancy cannot be allowed.
        if not self._push_down_locked:
            self._push_down_locked = True

            # by default, saves the parent value, which can later be used by any other object as necessary
            self.parent_value = parent_value

            # also allows custom operations
            self._capture_push_down_value(t, parent_value)

            if self.push_down:
                for e in self._causal_parameters:
                    e._push_down(t, parent_value)

            self._push_down_locked = False

    def _capture_push_down_value(self, t, parent_value):
        """
        Receives a value executed by a parent at the specified time and, if needed, stores this value locally.
        By default, actually nothing is stored, concrete subclasses must overload this method to do so.
        For example, events that are stateful and supposed to track some current value might store values
        produced by parents.

        :param t:
        :param parent_value:
        :return:
        """
        pass

    def _value_or_execute_if_event(self, var_name, x, t):
        if isinstance(x, Event):
            return x.execute(t)

        else:  # it is a scalar value
            return x

    def parallel_to(self, events):
        if self.parallel_events is None:
            self.parallel_events = []

        if isinstance(events, list):
            for pe in events:
                if isinstance(pe, Event):
                    self.parallel_events.append(pe)
        elif isinstance(events, Event):
            self.parallel_events.append(events)
        else:
            raise ValueError("Either a list of events or an event must be specified.")

        return self

    def value_at_pos(self, i):
        return self._generated_values[i]

    def value_at(self, t):
        return self._indexed_generated_values[t]

    def _default_name_if_none(self, name=None):
        if name is None:
            return f"{type(self).__name__}_{str(fresh_id())}"
        else:
            return name

    def reset(self):
        """
        Cleans all caches in order to allow the reuse of the event in a new generation process.
        :return: None
        """
        self._indexed_generated_values = {}
        self._generated_values = []
        self._last_pos = -1

        # clear causes too, if the object is not locked
        if not self._execution_locked:
            self._execution_locked = True

            for e in self._causal_parameters:
                e.reset()

            self._execution_locked = False

        self.parent_value = None

    def is_root_cause(self):
        """
        Checks whether this event depend on any other or not (i.e., it is a root cause).
        :return: True if no dependecies exist; False otherwise.
        """
        return len(self._causal_parameters) == 0

    def __str__(self):
        return self.name

    def __add__(self, other):
        return LambdaEvent(lambda t, i, mem, sub_events: sub_events['a'].execute(t) + sub_events['b'].execute(t),
                           sub_events={'a': self, 'b': other})

    def __sub__(self, other):
        return LambdaEvent(lambda t, i, mem, sub_events: sub_events['a'].execute(t) - sub_events['b'].execute(t),
                           sub_events={'a': self, 'b': other})

    def __mul__(self, other):
        return LambdaEvent(lambda t, i, mem, sub_events: sub_events['a'].execute(t) * sub_events['b'].execute(t),
                           sub_events={'a': self, 'b': other})

    def __truediv__(self, other):
        return LambdaEvent(lambda t, i, mem, sub_events: sub_events['a'].execute(t) / sub_events['b'].execute(t),
                           sub_events={'a': self, 'b': other})

    def _push_constants_down(self, scalar_values):
        """
        Recursively attributes the constants named in the specified dictionary to the appropriate events.

        :param scalar_values: A dict with the named values.
        :return:
        """
        # to avoid infinite loops in situations of mutual dependency, reentrancy cannot be allowed.
        if not self._push_down_locked:
            self._push_down_locked = True

            try:
                for event in self._causal_parameters:
                    if isinstance(event, ConstantEvent):
                        name = event.name
                        if name in scalar_values:
                            value = scalar_values[name]
                            # check whether the value is within the required bounds, and if not enforce it.
                            if event.require_lower_bound is not None and value < event.require_lower_bound:
                                event.constant = event.require_lower_bound

                            elif event.require_upper_bound is not None and value > event.require_upper_bound:
                                event.constant = event.require_upper_bound

                            else:
                                event.constant = value
            except KeyError as err:
                logging.debug(err)

            for event in self._causal_parameters:
                event._push_constants_down(scalar_values)

            self._push_down_locked = False

    def generalize_from_observations(self, observed_traces,
                                     n_simulations=20, max_optimization_evals=300,
                                     upper_bound=None, lower_bound=None,
                                     error_strategy='best_trace',
                                     error_metric='mae',
                                     generator=None,
                                     sample_proportion=1.0,
                                     verbose=False):
        """
        Given various observations, learn the model parameters that best fit them.

        :param observed_traces: A sequence of sequences of observations. E.g., [[1, 0, 1, 0, 1, 1], [1, 1, 0], ...];
                                or a sequence of Series indexed by timestamps.
        :param n_simulations: How many simulations per observed trace are to be performed when calculating the error.
        :param max_optimization_evals: How many passes to perform on the optimization procedure.
        :param lower_bound: If not otherwise given, this will be the lower bound of parameters being optimized.
        :param upper_bound: If not otherwise given, this will be the upper bound of parameters being optimized.
        :param error_strategy: The calculation strategy to use for the error function.
                               'best_trace' indicates that the error will consider the best trace only, ignoring
                               the others;
                               'all_traces' indicates that all traces will be considered equally.
        :param error_metric: How to measure the difference between two traces.
        :param generator: Specify the generator to consider when learning from data. If the observed traces
                          also include timestamps, this will force the learning mechanisms to align the temporal
                          indexes (date and time) for each point generated.
        :param sample_proportion: A float from 0.0 to 1.0 indicating how much of each trace should be used. This
                                  is useful to speed-up learning when traces are too large.
        :param verbose: Whether to print auxiliary information.
        :return:
        """

        # TODO factor the learning mechanism out, so that different mechanisms can be experimented with.

        def error(y_true, y_pred):
            if error_metric == 'mse':
                err_func = mean_squared_error
            elif error_metric == 'mae':
                err_func = mean_absolute_error
            elif error_metric == 'ks':
                err_func = lambda y_pred, y_true: ks_2samp(y_pred, y_true)[0]
            elif error_metric == 'cross-correlation':
                def aux_cross_corr(y_pred, y_true):
                    corr = signal.correlate(y_pred, y_true, mode='same')
                    return -sum(corr)

                err_func = aux_cross_corr
            else:
                raise ValueError(f"Invalid metric function: {error_metric}.")

            try:
                #print(f'Predicted vs True: {y_pred}, {y_true}')
                return err_func(y_pred=y_pred, y_true=y_true)
                # return mean_absolute_error(y_pred=y_pred, y_true=y_true)

            except ValueError:
                #print(f'Predicted vs True (under error): {y_pred}, {y_true}')
                y_pred[np.isinf(y_pred)] = LEARNING_CONFIG['large_default_value']
                y_pred[np.isnan(y_pred)] = LEARNING_CONFIG['large_default_value']

                y_true[np.isinf(y_true)] = LEARNING_CONFIG['large_default_value']
                y_true[np.isnan(y_true)] = LEARNING_CONFIG['large_default_value']

                return err_func(y_pred=y_pred, y_true=y_true)

        # objective function for black box optimization
        def aux_objective_function(args):
            try:
                # Change the constant parameters
                self._push_constants_down(args)

                #
                # Consider each observed trace in order to calculate the error. We'll do this in parallel.
                #

                # The function to run in parallel over each trace. It returns the error over that trace w.r.t. current
                # simulation parameters.
                def aux_trace_error(trace):
                    nonlocal generator
                    #
                    # calculate dates and temporal lengths
                    #
                    n_obs = len(trace)
                    if isinstance(trace, pd.Series):
                        trace_idx = trace.index
                        start_date = trace_idx[0]
                        end_date = trace_idx[-1]
                    else:
                        trace_idx = None
                        start_date = pd.Timestamp.today()
                        end_date = start_date + pd.offsets.Day(n_obs)

                    if generator is None:
                        generator = Generator(start_date=start_date, end_date=end_date)


                    #
                    # run simulation a few times
                    #
                    sim_outputs = []
                    trace_segments = []
                    for i in range(0, n_simulations):

                        # If requested, use only part of the available data.
                        if sample_proportion < 1.0:
                            sim_index = pd.date_range(generator.start_date, generator.end_date, freq=generator.freq)
                            sim_length = math.ceil(len(sim_index)*sample_proportion)
                            sim_start_pos = random.randint(0, len(sim_index)-1)
                            sim_index = sim_index[sim_start_pos:sim_start_pos + sim_length]
                            sim_start_date = sim_index[0]
                            sim_end_date = sim_index[-1]
                        else:
                            sim_start_date = generator.start_date
                            sim_end_date = generator.end_date

                        res = generator.generate(self, new_start_date=sim_start_date, new_end_date=sim_end_date)

                        # is the generated index included in the observed trace index?
                        if trace_idx is not None:
                            # ensure the simulated index is within the observed trace one ...
                            if not res.index.isin(trace_idx):
                                raise ValueError("When the observed trace contains a temporal index, the simulated"
                                                 " index must be included therein.")

                            # ... and then only keep the intersection points for comparison. This will
                            # align the _dates_ of the values when available.
                            trace_segment = trace[trace.index.isin(res.index)]
                            trace_segments.append(trace_segment.values)

                        else:
                            trace_segment = trace[:len(res)]
                            trace_segments.append(trace_segment)

                        sim_outputs.append(res.values[:len(trace_segment), :])

                    #
                    # calculate error in relation to observations
                    #

                    sim_outputs_flattened = functools.reduce(lambda x, y: np.concatenate([x, y],
                                                                                         axis=None),
                                                             sim_outputs)
                    trace_copies_flattened = functools.reduce(lambda x, y: np.concatenate([x, y],
                                                                                          axis=None),
                                                              trace_segments)
                    return error(trace_copies_flattened,
                                 sim_outputs_flattened)

                #errors = [aux_trace_error(trace) for trace in rescaled_traces]
                errors = Parallel(n_jobs=-2)(delayed(aux_trace_error)(trace) for trace in observed_traces)

                # decide how to compute the final error. Focus on specific traces or consider all of them?
                if error_strategy == 'best_trace':
                    err = min(errors)  # selects the error w.r.t. the best trace
                    min_error_trace_pos = np.argmin(errors)
                elif error_strategy == 'all_traces':
                    err = np.mean(errors)
                    min_error_trace_pos = None
                else:
                    raise ValueError(f"Invalid error_strategy: {error_strategy}.")

                return {'loss': err, 'status': STATUS_OK, 'min_error_trace_pos': min_error_trace_pos}

            except AssertionError as ae:
                return {'status': STATUS_FAIL}

            except ValueError as ve:
                return {'status': STATUS_FAIL}

        params = self._causal_parameters_closure(only_learnable=True)
        # for p in params:
        #    print(p)

        #
        # define parameter search space
        #
        space = {}
        for param in params:
            if isinstance(param, ConstantEvent):
                # acquire upper bound
                if param.require_upper_bound is not None:
                    ub = param.require_upper_bound
                    # print(f"Upper bound constraint found: {ub}")
                else:
                    ub = upper_bound

                # acquire lower bound
                if param.require_lower_bound is not None:
                    lb = param.require_lower_bound
                    # print(f"Lower bound constraint found: {lb}")
                else:
                    lb = lower_bound

                # the actual random variables
                if (lb is not None) and (ub is not None) and \
                        (param.learning_normal_mean is None) and (param.learning_normal_std is None):
                    # if we have both bounds and nothing else, let's use them
                    space[param.name] = hyperopt.hp.uniform(param.name, lb, ub)

                else:
                    normal_mean = \
                        param.learning_normal_mean if param.learning_normal_mean is not None else param.constant
                    normal_std = \
                        abs(param.learning_normal_std if param.learning_normal_std is not None else normal_mean)

                    space[param.name] = hyperopt.hp.normal(param.name, mu=param.constant, sigma=max(normal_std, 1))
                    # TODO somehow also enforce upper and lower bounds here using some hyperopt mechanism

        if verbose:
            print(f"Considering {len(space)} variables.")

        # optimize
        trials = hyperopt.Trials()
        best = hyperopt.fmin(aux_objective_function, space, algo=hyperopt.tpe.suggest,
                             max_evals=max_optimization_evals, trials=trials)
        #   hyperopt.tpe.suggest
        #   hyperopt.anneal.suggest
        #   hyperopt.atpe.suggest
        #   hyperopt.rand.suggest

        if verbose:
            print(best)
            print(f'Best trial = {trials.best_trial}')

        best_min_error_trace_pos = trials.best_trial['result']['min_error_trace_pos']

        # propagate the learned parameters down
        self._push_constants_down(best)

    def clone(self):
        """
        Produces a copy of the present object, ensuring that elements that must be unique or shared are indeed so.

        :return: A clone of the object.
        """

        # custom implementation of __deepcopy__ ensure that names are actually unique, among other details.
        return copy.deepcopy(self)

    def _fix_copy(self, c):
        # adjust elements that must be unique or have references preserved
        c.name = f"{self.name}_clone-{str(fresh_id())}"
        c.parallel_events = self.parallel_events

        return c

    def __copy__(self):
        # adapted from
        # https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object

        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)

        self._fix_copy(result)

        return result

    def __deepcopy__(self, memo):
        # adapted from
        # https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object

        if id(self) not in memo:
            cls = self.__class__
            result = cls.__new__(cls)

            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            self._fix_copy(result)
        else:
            result = memo[id(self)]

        return result


###############################################################################
# More core events
###############################################################################
class ConstantEvent(Event):
    def __init__(self, constant=0.0,
                 require_lower_bound=None,
                 require_upper_bound=None,
                 learning_normal_mean=None,
                 learning_normal_std=None,
                 name=None, parallel_events=None, push_down=True, allow_learning=True):

        super().__init__(name, parallel_events, push_down, allow_learning)
        self.constant = constant
        self.require_lower_bound = require_lower_bound
        self.require_upper_bound = require_upper_bound

        self.learning_normal_mean = learning_normal_mean
        self.learning_normal_std = learning_normal_std

        self._check_constraints()

    def _execute(self, t, i):
        self._check_constraints()
        return self._value_or_execute_if_event('constant', self.constant, t)

    def _check_constraints(self):
        if (self.require_lower_bound is not None) and (self.constant < self.require_lower_bound):
            raise AssertionError(f"Constraint violation: constant must be positive, but was {self.constant}.")

        if (self.require_upper_bound is not None) and (self.constant > self.require_upper_bound):
            raise AssertionError(f"Constraint violation: constant must be less than or equal to the upper bound "
                                 f"{self.require_upper_bound}, but was {self.constant}.")


class LambdaEvent(Event):

    def __init__(self, func, sub_events={}, name=None, parallel_events=None, push_down=True, allow_learning=True):
        super().__init__(name, parallel_events, push_down, allow_learning)
        self.func = func
        self.sub_events = sub_events

        self.memory = {}

        self._make_sub_events_causal(sub_events)

        if self.func.__closure__ is not None:
            raise ValueError(f"The specified function cannot depend on free variables. "
                             f"Its closure must be empty, but was {self.func.__closure__} instead!")

    def _make_sub_events_causal(self, sub_events):
        if sub_events is not None:
            for key, se in sub_events.items():
                if isinstance(se, Event):
                    self._causal_parameters.append(se)

    def _execute(self, t, i):
        return self.func(t, i, self.memory, self.sub_events)

    def reset(self):
        self.memory = {}
        super().reset()

    def __copy__(self):
        result = LambdaEvent(func=self.func, sub_events=self.sub_events,
                             name=self.name, parallel_events=self.parallel_events, push_down=self.push_down,
                             allow_learning=self._allow_learning)

        self._fix_copy(result)

        return result

    def __deepcopy__(self, memo):
        if id(self) not in memo:
            result = LambdaEvent(func=self.func, sub_events={},
                                 name=self.name, parallel_events=self.parallel_events, push_down=self.push_down,
                                 allow_learning=self._allow_learning)

            # to handle circular references, we need to first update the memo dict with the new reference
            # for the present object before deepcopying its attributes.
            memo[id(self)] = result
            sub_events_deepcopy = copy.deepcopy(self.sub_events, memo)

            # update deepcopied attributes
            result.sub_events = sub_events_deepcopy
            result._init_causal_parameters()
            result._make_sub_events_causal(sub_events_deepcopy)

            self._fix_copy(result)
        else:
            result = memo[id(self)]

        return result


###############################################################################
# Generating mechanisms and other related conveniences
###############################################################################
class Generator:

    def __init__(self, start_date, end_date, between_time=None, freq='D', filter_func=lambda df: df):
        """
        Creates a new time series generator using the specified events. The names of the events will be used
        later to name the columns of generated DataFrames.

        :param start_date: The first date of the series to generate.
        :param end_date: The last date of the series to generate.
        :param between_time A pair which, when specified, define a start and end times for every day generated
        :param freq: The frequency to be used, in terms of Pandas frequency strings.
        :param filter_func: A filter function to apply to the generated data before returning.
        """
        self.between_time = between_time
        self.freq = freq
        self.filter_func = filter_func
        self.start_date = start_date
        self.end_date = end_date

    def generate(self, events, n=1, new_start_date=None, new_end_date=None):
        """
        Generates time series from the model assigned to the present generator.

        :param events: Either a list of events or a dict of events. In the former case, the name of each
        event is retrieved from the event itself. In the latter case, the user can specify new names.
        :param n: The number of series to generate.
        :param new_start_date: If specified, overrides the class' start_date for this particular generation.
        :param new_end_date: If specified, overrides the class' end_date for this particular generation.

        :return: A list of generated time series.
        """

        #
        # Setup events to use
        #
        self._set_events(events)

        # calculate proper start data
        if new_start_date is not None:
            start = new_start_date
        else:
            start = self.start_date

        # calculate proper end date
        if new_end_date is not None:
            end = new_end_date
        else:
            end = self.end_date


        #
        # Generate data from the given events.
        #
        generated_data = []
        for i in range(0, n):
            values = {}
            dates = pd.date_range(start, end, freq=self.freq)
            for name in self.named_events:
                self.named_events[name].reset()  # clears the cache

            for t in dates:
                for name in self.named_events:
                    if name not in values:
                        values[name] = []

                    values[name].append(self.named_events[name].execute(t))

            df = pd.DataFrame(values, index=dates)
            if self.between_time is not None:
                df = df.between_time(self.between_time[0], self.between_time[1])

            df = self.filter_func(df)
            generated_data.append(df)

        if len(generated_data) > 1:
            return generated_data
        else:
            return generated_data[0]

    def _set_events(self, events):
        if isinstance(events, dict):
            self.named_events = events
        else:
            self.named_events = {}

            # make iterable
            if not isinstance(events, list):
                events = [events]

            for i, event in enumerate(events):
                if event.name is not None:
                    self.named_events[event.name] = event
                else:
                    self.named_events['Event ' + str(i)] = event


def generate(model, start_date, end_date, n=1, freq='D', filter_func=lambda df: df):
    """
    A convenience method to generate time series from the specified model using the default generator.

    :param model: The model from which the data is to be generated.
    :param start_date: The first date of the series to generate.
    :param end_date: The last date of the series to generate.
    :param n: The number of series to generate.
    :param freq: The frequency to be used, in terms of Pandas frequency strings.
    :param filter_func: A filter function to apply to the generated data before returning.

    :return: A list of generated time series.
    """
    data = Generator(start_date=start_date,
                     end_date=end_date,
                     freq=freq,
                     filter_func=filter_func) \
        .generate(model, n=n)

    return data


def generate_and_plot(model, start_date, end_date, n=1, freq='D', return_data=False, filter_func=lambda df: df,
                      grid=True):
    """
     A convenience method to generate time series from the specified model using the default generator, and also plot
     them.

     :param model: The model from which the data is to be generated.
     :param start_date: The first date of the series to generate.
     :param end_date: The last date of the series to generate.
     :param n: The number of series to generate.
     :param freq: The frequency to be used, in terms of Pandas frequency strings.
     :param return_data: Whether to return the generated data or not. The default is False, useful when
                         usin Jupyter notebooks only to show the charts, without further data processing.
     :param filter_func: A filter function to apply to the generated data before returning.

     :return: A list of generated time series.
     """
    data = generate(model, start_date, end_date, n=n, freq=freq, filter_func=filter_func)

    # plot
    def aux_plot(df):
        df.plot(grid=grid)

    for i in range(0, n):
        aux_plot(data[i]) if n > 1 else aux_plot(data)
        plt.show()

    if return_data:
        return data


def save_to_csv(data, common_name, **kwargs):
    for i, df in enumerate(data):
        name = common_name + '_' + i
        df.to_csv(name, kwargs)
