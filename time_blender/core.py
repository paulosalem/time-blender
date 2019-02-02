import logging
import random

import pandas as pd

from time_blender.util import fresh_id

import pymc3 as pm


class Event:

    def __init__(self, name=None, parallel_events=None, push_down=False):
        self.name = name

        self._indexed_generated_values = {} # time to value
        self._generated_values = [] # sequence of values
        self._last_pos = -1

        # determine which attributes are causal events
        self._causal_parameters = []
        for k, v in self.__dict__.items():
            if isinstance(v, Event):
                self._causal_parameters.append(v)

        if parallel_events is not None:
            self.parallel_events = []
            self.parallel_to(parallel_events)
        else:
            self.parallel_events = None

        self.push_down = push_down

        self._execution_locked = False

    def execute(self, t):
        """
        Executes the event and generates an output for the present moment.

        :param t: The time in which the event takes place.
        :return: The scalar value of the event in the specified moment.
        """

        if not self._execution_locked:
            self._execution_locked = True

            # update parallel events
            if self.parallel_events is not None:
                for e in self.parallel_events:
                    e.execute(t)

            # process this event
            if t not in self._indexed_generated_values:
                self._last_pos += 1
                # save the value for future reference and to avoid recomputing
                v = self._execute(t, self._last_pos)
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

    def _value_or_execute_if_event(self, x, t):
        if isinstance(x, Event):
            return x.execute(t)
        else:
            return x

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

    def _pymc3_model_variables(self, model, observations=None):
        found_random_event = False
        for event in self._causal_parameters:
            if not found_random_event:
                event._pymc3_model_variables(model, observations)
                if isinstance(event, RandomEvent):
                    found_random_event = True
            else:
                # There can be only one root RandomEvent because observations must be related to exactly one
                # event in the causal graph.
                raise ValueError("There can be only one root random event (all other random events must be causally \
                                  linked to it)")

    def _pymc3_push_distributions_down(self, trace):
        for event in self._causal_parameters:
            event._pymc3_push_distributions_down(trace)

    def generalize_from_observations(self, observations):
        """
        Given various observations, learn the model parameters that best fit them.

        :param observations: A sequence observations. E.g., [1, 0, 1, 0, 1, 1].
        :return:
        """
        raise NotImplementedError("Only random events can be subject to learning.")

    def is_root_cause(self):
        """
        Checks whether this event depend on any other or not (i.e., it is a root cause).
        :return: True if no dependecies exist; False otherwise.
        """
        return len(self._causal_parameters) == 0

    def __add__(self, other):
            return LambdaEvent(lambda t, i, mem: self.execute(t) + other.execute(t), sub_events=[other])

    def __sub__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) - other.execute(t), sub_events=[other])

    def __mul__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) * other.execute(t), sub_events=[other])

    def __truediv__(self, other):
        return LambdaEvent(lambda t, i, mem: self.execute(t) / other.execute(t), sub_events=[other])


class RandomEvent(Event):

    def __init__(self, pymc3_distribution_func, name=None, parallel_events=None, push_down=False):
        self.pymc3_distribution_func = pymc3_distribution_func
        self.distribution_sample = []
        super().__init__(name, parallel_events, push_down)

    def _pymc3_model_variables(self, model, observations=None):
        return self.pymc3_distribution_func(model, observations)

    @staticmethod
    def _pymc3_model_variables_if_distribution(model, x, observations=None):
        if isinstance(x, RandomEvent):
            return x._pymc3_model_variables(model, observations)
        elif isinstance(x, Event):
            raise ValueError("Only random events or scalars are allowed.")
        else:
            return x

    def _pymc3_push_distributions_down(self, trace):
        try:
            self.distribution_sample = trace[self.name]
        except KeyError as err:
            logging.debug(err)

        super()._pymc3_push_distributions_down(trace)

    def generalize_from_observations(self, observations):
        """
        Given various observations, learn the model parameters that best fit them.

        :param observations: A sequence observations. E.g., [1, 0, 1, 0, 1, 1].
        :return:
        """
        model = pm.Model()
        with model:
            self._pymc3_model_variables(model, observations)

            # Calculate a posteriori distribution from observations
            step = pm.Metropolis()
            self.trace = pm.sample(10000, tune=5000, step=step) # TODO customize parameters and algorithm

            # propagate the new distributions down
            self._pymc3_push_distributions_down(self.trace)

    def sample_from_learned_distribution(self):
        """
        Returns a sample based on the (a posteriori) learned distribution.
        :return: a scalar value.
        """
        if len(self.distribution_sample) == 0:
            raise ValueError("Cannot pick from empty distribution sample.")

        return random.choice(self.distribution_sample)

    def sample_from_definition(self, t):
        raise NotImplementedError("Must be implemented by concrete subclasses.")

    def has_learned(self):
        """
        Checks whether this event has been subject to learning from data.
        :return: True if learning took place; False otherwise.
        """
        return len(self.distribution_sample) > 0

    def _should_use_learned_sample(self):
        return self.is_root_cause() and self.has_learned()

    def _execute(self, t, i):
        if self._should_use_learned_sample():
            return self.sample_from_learned_distribution()
        else:
            return self.sample_from_definition(t)


class LambdaEvent(Event):

    def __init__(self, func, sub_events=[], name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.func = func
        self.memory = {}

        for se in sub_events:
            if isinstance(se, Event):
                self._causal_parameters.append(se)

    def _execute(self, t, i):
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
