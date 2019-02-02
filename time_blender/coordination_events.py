from time_blender.core import Event, RandomEvent
import time_blender.config as config
import pymc3 as pm
import random
import numpy as np

# Filters, connectors, etc.


# TODO DatetimeFilter  # weekends, specific hours, holidays, etc.

from time_blender.random_events import NormalEvent
from time_blender.util import PyMC3Utils


class Once(Event):
    """
    Executes and memoizes the result of an event, so that it is not recomputed later.
    """

    def __init__(self, event, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.event = event
        self.value = None

    def _execute(self, t, i):
        if self.value is None:
            self.value = self.event.execute(t)

        return self.value

    def reset(self):
        super().reset()
        self.value = None


class PastEvent(Event):
    def __init__(self, delay, undefined_value=0.0, refers_to=None, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.delay = delay
        self.undefined_value = undefined_value
        self.event = refers_to

    def refers_to(self, event):
        self.event = event
        return self

    def _execute(self, t, i):
        if self.event is None:
            raise ValueError("The event to which the present event refers to has not been defined yet.")
        else:
            pos = i - self.delay
            if pos >= 0:
                try:
                    v = self.event.value_at_pos(pos)

                except IndexError:
                    self.event.execute(t)
                    v = self.event.value_at_pos(pos)

                return v
            else:
                return self.undefined_value


class SeasonalEvent(Event):
    def __init__(self, event, base=0, year:int=None, month:int=None, day:int=None,
                 hour:int=None, minute:int=None, second:int=None,
                 name=None, parallel_events=None, push_down=False):

        self.event = event
        self.base = base
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i):

        def aux_match(a, b, cont):

            if a is not None:
                if a == b:
                    return cont
                else:
                    return False
            else:
                return cont

        is_season = \
            aux_match(self.year, t.year,
                      aux_match(self.month, t.month,
                                aux_match(self.day, t.day,
                                          aux_match(self.hour, t.hour,
                                                    aux_match(self.minute, t.minute,
                                                              aux_match(self.second, t.second,
                                                                        True))))))
        if is_season:
            return self.event.execute(t)
        else:
            return self.base


class Choice(RandomEvent):

    def __init__(self, events, name=None, parallel_events=None, push_down=False):
        """

        :param events: When given a list of events, a uniform distribution will be assumed. When given a dict,
                       keys are events and values are their probabilities according to a categorical distribution.
        :param name:
        """

        if isinstance(events, dict):
            self.events = list(events.keys())
            self.probs = list(events.values())
            self.events_probs = events

        elif isinstance(events, list):
            self.events = events

            uni_prob = 1.0/len(events)
            self.events_probs = {e: uni_prob for e in events}
            self.probs = list(self.events_probs.values())

        name = self._default_name_if_none(name)

        # a function to specify the PyMC3 structure to use for choice
        def aux(model, obs):
            c = pm.Categorical(name, p=self.probs, observed=obs)
            all_pymc3_vars = [self._pymc3_model_variables_if_distribution(model, e) for e in events]
            return PyMC3Utils.multi_switch(all_pymc3_vars, c)

        super().__init__(aux, name, parallel_events, push_down)

    def sample_from_definition(self, t):
        raise NotImplementedError("Choice cannot be sampled directly from.")

    def sample_from_learned_distribution(self):
        raise NotImplementedError("Choice cannot be sampled directly from.")

    def choose(self, t):
        if self._should_use_learned_sample():
            idx = self.sample_from_learned_distribution()
        else:
            d = pm.Categorical.dist(p=self.probs)
            idx = d.random()

        return self.events[idx]

    def _execute(self, t, i):
        return self.choose(t).execute(t)


class Piecewise(Event):

    def __init__(self, events: list, t_separators:list=None,
                 name=None, parallel_events=None, push_down=False):

        self.events = events
        self.event_choice = Choice(events)
        self.t_separators = t_separators

        self._cur_separator_pos = 0
        self._cur_separator = None

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i):
        if self._cur_separator_pos < len(self.t_separators):
            self._cur_separator = self.t_separators[self._cur_separator_pos]

            if i >= self._value_or_execute_if_event(self._cur_separator, t):
                self._cur_separator_pos += 1

        return self.events[self._cur_separator_pos].execute(t)


# TODO  class RandomPiecewise(RandomEvent):
#
# def __init__(self, events: list, t_separators:list=None, spacing_mean:float=None, spacing_sd:float=None, mode: str='fixed',
#                  name=None, parallel_events=None, random_seed=None):


class Top(RandomEvent):

    def __init__(self, event, resistance_probability, name=None, parallel_events=None, push_down=False):
        self.event = event
        self.resistance_probability = resistance_probability

        super().__init__(name, parallel_events, push_down)

# Trading: tops, bottoms, panics, shocks, news, etc.
# TODO

# Mean-reverting
# TODO
