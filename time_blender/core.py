import logging
import random
import copy

import numpy as np
import pandas as pd
import functools
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hyperopt
from joblib import Parallel, delayed

from time_blender.util import fresh_id


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
    if not isinstance(value, Event):
        if isinstance(value, int) or isinstance(value, float):
            value = ConstantEvent(value, name=f'{prefix}_{name}', **kwargs)

    return value


###############################################################################
# Core event classes
###############################################################################
class Event:

    def __init__(self, name=None, parallel_events=None, push_down=False):
        self.name = self._default_name_if_none(name)

        self._indexed_generated_values = {} # time to value
        self._generated_values = [] # sequence of values
        self._last_pos = -1

        # determine which attributes are causal events
        self._init_causal_parameters()

        # set parallel events
        if parallel_events is not None:
            self.parallel_events = []
            self.parallel_to(parallel_events)
        else:
            self.parallel_events = None

        self.push_down = push_down

        self._execution_locked = False

        #
        # random variables after sampling
        #
        self.distribution_sample = []  # TODO obsolete?

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

    def _causal_parameters_closure(self):
        closure = []
        for event in self._causal_parameters:
            closure.append(event)
            closure = closure + event._causal_parameters_closure()

        return closure

    def execute(self, t, obs=None):
        """
        Executes the event and generates an output for the present moment.

        :param obs:
        :param t: The time in which the event takes place.
        :return: The scalar value of the event in the specified moment.
        """

        # TODO check obs parameters validity?

        if not self._execution_locked:
            self._execution_locked = True

            # update parallel events
            if self.parallel_events is not None:
                for e in self.parallel_events:
                    e.execute(t, obs=obs)

            # process this event
            if t not in self._indexed_generated_values:
                self._last_pos += 1

                # should we use previously recorded (learned) values or re-execute?
                if self._should_use_learned_sample():
                    v = self.sample_from_learned_distribution()
                else:
                    v = self._execute(t, self._last_pos, obs=obs)

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

    def _execute(self, t, i, obs=None):
        raise NotImplementedError("Must be implemented by concrete subclasses.")

    def _push_down(self, t, parent_value):
        """
        Given a value executed at the specified moment by a parent event, pushes it down to its children
        events. That is to say, provides a downward path for executed values, opposite to the the regular
        information flow from children to parent.

        :param t: The execution moment.
        :param parent_value: The value executed by a parent.
        :return: None
        """
        self._capture_push_down_value(t, parent_value)

        if self.push_down:
            for e in self._causal_parameters:
                e._push_down(t, parent_value)

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

    def _value_or_execute_if_event(self, var_name, x, t,  obs=None):
        if isinstance(x, Event):
            return x.execute(t, obs=obs)

        else:  # it is a scalar value
            return x

    def _temporal_name(self, t):
        """
        Returns the event's name augmented with the specified instant.
        :param t:
        :return:
        """
        return f'{self.name}_t{t}'

    def parallel_to(self, events):
        if self.parallel_events is None:
            self.parallel_events = []

        if isinstance(events, list):
            for pe in events:
                if isinstance(pe, Event):
                    self.parallel_events.append(pe)
                    self._causal_parameters.append(pe)
        elif isinstance(events, Event):
            self.parallel_events.append(events)
            self._causal_parameters.append(events)
        else:
            raise ValueError("Either a list of events or an event must be specified.")

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

    def is_root_cause(self):
        """
        Checks whether this event depend on any other or not (i.e., it is a root cause).
        :return: True if no dependecies exist; False otherwise.
        """
        return len(self._causal_parameters) == 0

    def __str__(self):
        return self.name

    def __add__(self, other):
            return LambdaEvent(lambda t, i, mem: self.execute(t) + other.execute(t), sub_events=[other])

    def __sub__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) - other.execute(t), sub_events=[other])

    def __mul__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) * other.execute(t), sub_events=[other])

    def __truediv__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) / other.execute(t), sub_events=[other])

    def _push_constants_down(self, scalar_values):
        """
        Recursively attributes the constants named in the specified dictionary to the appropriate events.

        :param scalar_values: A dict with the named values.
        :return:
        """
        try:
            for event in self._causal_parameters:
                if isinstance(event, ConstantEvent):
                    name = event.name
                    if name in scalar_values:
                        event.constant = scalar_values[name]

        except KeyError as err:
            logging.debug(err)

        for event in self._causal_parameters:
            event._push_constants_down(scalar_values)

    def generalize_from_observations(self, observed_traces,
                                     n_simulations=20, max_optimization_evals=300,
                                     upper_bound=20, lower_bound=-20,
                                     error_strategy='best_trace'):
        """
        Given various observations, learn the model parameters that best fit them.

        :param observed_traces: A sequence of sequences of observations. E.g., [[1, 0, 1, 0, 1, 1], [1, 1, 0], ...].
        :param n_simulations: How many simulations per observed trace are to be performed when calculating the error.
        :param max_optimization_evals: How many passes to perform on the optimization procedure.
        :param lower_bound: If not otherwise given, this will be the lower bound of parameters being optimized.
        :param upper_bound: If not otherwise given, this will be the upper bound of parameters being optimized.
        :param error_strategy: The calculation strategy to use for the error function.
                               'best_trace' indicates that the error will consider the best trace only, ignoring
                               the others;
                               'all_traces' indicates that all traces will be considered equally.
        :return:
        """
        def error(y_true, y_pred):
            #return mean_squared_error(y_pred=y_pred, y_true=y_true)
            return mean_absolute_error(y_pred=y_pred, y_true=y_true)
        
        # objective function for black box optimization
        def aux_objective_function(args):
            # Change the constant parameters
            self._push_constants_down(args)

            #
            # Consider each observed trace in order to calculate the error. We'll do this in parallel.
            #

            # The function to run in parallel over each trace. It returns the error over that trace w.r.t. current
            # simulation parameters.
            def aux_trace_error(trace):
                #
                # calculate dates and temporal lengths
                #
                n_steps = len(trace)
                start_date = pd.Timestamp.today()
                end_date = start_date + pd.offsets.Day(n_steps)
                n_obs = len(trace)

                #
                # run simulation a few times
                #
                generator = Generator(self)
                sim_outputs = []
                ###trace_diffs = []
                for i in range(0, n_simulations):
                    res = generator.generate(start_date=start_date, end_date=end_date)
                    sim_outputs.append(res[0].values[:n_obs, :])

                #
                # calculate error in relation to observations
                #

                sim_outputs_flattened = functools.reduce(lambda x, y: np.concatenate([x, y],
                                                                                     axis=None),
                                                         sim_outputs)
                trace_copies_flattened = np.concatenate([trace for i in range(0, len(sim_outputs))], axis=None)

                return error(trace_copies_flattened, sim_outputs_flattened)

            # Call the trace processing function in parallel. n_jobs=-2 means that all CPUs minus one are used.
            errors = Parallel(n_jobs=-2)(delayed(aux_trace_error)(trace) for trace in observed_traces)

            # decide how to compute the final error. Focus on specific traces or consider all of them?
            if error_strategy == 'best_trace':
                err = min(errors)  # selects the error w.r.t. the best trace
            elif error_strategy == 'all_traces':
                err = np.mean(errors)
            else:
                raise ValueError(f"Invalid error_strategy: {error_strategy}.")

            return err

        params = self._causal_parameters_closure()
        for p in params:
            print(p)

        # define parameter search space
        space = {}
        for param in params:
            if isinstance(param, ConstantEvent):
                # check upper bound
                if param.require_upper_bound is not None:
                    ub = param.require_upper_bound
                    #print(f"Upper bound constraint found: {ub}")
                else:
                    ub = upper_bound

                # check positivity constraint
                if param.require_lower_bound is not None:
                    lb = param.require_lower_bound
                    #print(f"Lower bound constraint found: {lb}")
                else:
                    lb = lower_bound

                space[param.name] = hyperopt.hp.uniform(param.name, lb, ub)

        print(space)

        # optimize
        trials = hyperopt.Trials()
        best = hyperopt.fmin(aux_objective_function, space, algo=hyperopt.tpe.suggest,
                             max_evals=max_optimization_evals, trials=trials)
        #   hyperopt.tpe.suggest
        #   hyperopt.anneal.suggest
        #   hyperopt.atpe.suggest
        #   hyperopt.rand.suggest

        print(best)

        # propagate the learned parameters down
        self._push_constants_down(best)

    def sample_from_learned_distribution(self):
        """
        Returns a sample based on the (a posteriori) learned distribution.
        :return: a scalar value.
        """
        if len(self.distribution_sample) == 0:
            raise ValueError("Cannot pick from empty distribution sample.")

        return random.choice(self.distribution_sample)

    def has_learned(self):
        """
        Checks whether this event has been subject to learning from data.
        :return: True if learning took place; False otherwise.
        """
        return len(self.distribution_sample) > 0

    def _should_use_learned_sample(self):
        return self.is_root_cause() and self.has_learned() # TODO insist in root cause?

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

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        self._fix_copy(result)
        return result


class ConstantEvent(Event):
    def __init__(self, constant, require_lower_bound=None, require_upper_bound=None,
                 name=None, parallel_events=None, push_down=False):

        super().__init__(name, parallel_events, push_down)
        self.constant = constant
        self.require_lower_bound = require_lower_bound
        self.require_upper_bound = require_upper_bound

        self._check_constraints()

    def _execute(self, t, i, obs=None):
        self._check_constraints()
        return self._value_or_execute_if_event('constant', self.constant, t)

    def _check_constraints(self):
        if (self.require_lower_bound is not None) and (self.constant < self.require_lower_bound):
            raise Exception(f"Constraint violation: constant must be positive, but was {self.constant}.")

        if (self.require_upper_bound is not None) and (self.constant > self.require_upper_bound):
            raise Exception(f"Constraint violation: constant must be less than or equal to the upper bound "
                            f"{self.require_upper_bound}, but was {self.constant}.")


class LambdaEvent(Event):

    def __init__(self, func, sub_events=None, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.func = func
        self.memory = {}

        if sub_events is not None:
            for se in sub_events:
                if isinstance(se, Event):
                    self._causal_parameters.append(se)

    def _execute(self, t, i, obs=None):
        return self.func(t, i, self.memory)

    def reset(self):
        self.memory = {}
        super().reset()


class Generator:

    def __init__(self, events):
        """
        Creates a new time series generator using the specified events. The names of the events will be used
        later to name the columns of generated DataFrames.

        :param events: Either a list of events or a dict of events. In the former case, the name of each
        event is retrieved from the event itself. In the latter case, the user can specifiy new names.
        """
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

    def generate(self, start_date, end_date, n=1, between_time=None, freq='D', filter_func=lambda df: df):
        """
        Generates time series from the model assigned to the present generator.

        :param start_date: The first date of the series to generate.
        :param end_date: The last date of the series to generate.
        :param n: The number of series to generate.
        :param freq: The frequency to be used, in terms of Pandas frequency strings.
        :param filter_func: A filter function to apply to the generated data before returning.

        :return: A list of generated time series.
        """
        generated_data = []
        for i in range(0, n):
            values = {}
            dates = pd.date_range(start_date, end_date, freq=freq)
            for name in self.named_events:
                self.named_events[name].reset()  # clears the cache

            for t in dates:
                for name in self.named_events:
                    if name not in values:
                        values[name] = []

                    values[name].append(self.named_events[name].execute(t))

            df = pd.DataFrame(values, index=dates)
            if between_time is not None:
                df = df.between_time(between_time[0], between_time[1])

            df = filter_func(df)
            generated_data.append(df)

        return generated_data


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
    data = Generator(model).generate(start_date,
                                     end_date,
                                     n=n,
                                     freq=freq,
                                     filter_func=filter_func)

    return data


def save_to_csv(data, common_name, **kwargs):
    for i, df in enumerate(data):
        name = common_name + '_' + i
        df.to_csv(name, kwargs)
