from time_blender.core import Event
import time_blender.config as config
import random
import numpy as np
from numpy.random import choice

# Filters, connectors, etc.
from time_blender.random_events import BernoulliEvent
from time_blender.util import is_sequence


class OnceEvent(Event):
    """
    Executes and memoizes the result of an event, so that it is not recomputed later.
    """

    def __init__(self, event, name=None, parallel_events=None, push_down=True, allow_learning=True):
        self.event = event
        self.value = None
        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        if self.value is None:
            self.value = self.event.execute(t)

        return self.value

    def reset(self):
        super().reset()
        self.value = None


class ParentValueEvent(Event):

    def __init__(self, event=None, default=0.0, name=None, parallel_events=None, push_down=True, allow_learning=False):
        self.event = event
        self.default = self._wrapped_param(name, 'default', default)

        self.value = None
        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        if self.event is not None:
            self.event.execute(t)
            self.value = self.event.parent_value
        else:
            # if self.event is None, it means this event's parent value is desired.
            if self.parent_value is not None:
                self.value = self.parent_value
            else:
                # if it is the first execution, there's no parent_value yet.
                self.value = self.default.execute(t)

        return self.value

    def reset(self):
        self.value = None
        super().reset()


class CumulativeEvent(Event):
    """
    Executes and accumulates the result of an underlying event over time, so that the result is the cumulative sum
    of that event.
    """

    def __init__(self, event, name=None, parallel_events=None, push_down=True, allow_learning=True,
                 capture_parent_value=True):
        """

        :param event:
        :param name:
        :param parallel_events:
        :param push_down:
        :param allow_learning:
        :param capture_parent_value: Whether the parent value should be used as the new current value to which
                                     the event's execution is added. This is useful to embed the present event
                                     into larger contexts and accumulate on top of their feedback.
        """
        self.event = event
        self.value = None

        self.capture_parent_value = capture_parent_value

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        if self.value is None:
            # the first result in the series
            self.value = self.event.execute(t)
        else:
            # accumulates with past results
            self.value = self.value + self.event.execute(t)

        return self.value

    def _capture_push_down_value(self, t, parent_value):
        if self.capture_parent_value:
            self.value = parent_value

    def reset(self):
        super().reset()
        self.value = None


class PastEvent(Event):
    """
    Refers to a past event considering a certain delay. Typically, we don't want to make the delay parameter
    learnable, so it is recommended to set allow_learning=False in the constructor.
    """
    def __init__(self, delay, undefined_value=0.0, refers_to=None, name=None, parallel_events=None, push_down=True,
                 allow_learning=False):

        self.delay = self._wrapped_param(name, 'delay', delay)
        self.undefined_value = self._wrapped_param(name, 'undefined_value', undefined_value)
        self.event = refers_to

        super().__init__(name, parallel_events, push_down, allow_learning)

    def refers_to(self, event):
        self.event = event
        self._init_causal_parameters()
        return self

    def _execute(self, t, i):
        if self.event is None:
            raise ValueError("The event to which the present event refers to has not been defined yet.")
        else:
            pos = int(i - self.delay.execute(t))
            if pos >= 0:
                try:
                    v = self.event.value_at_pos(pos)

                except IndexError:
                    self.event.execute(t)
                    v = self.event.value_at_pos(pos)

                except Exception:
                    raise ValueError(f'Invalid position for past event: {pos}.')

                return v
            else:
                return self.undefined_value.constant


class ReferenceEvent(PastEvent):
    def __init__(self, undefined_value=0.0, refers_to=None, name=None, parallel_events=None, push_down=True,
                 allow_learning=False):

        super().__init__(delay=0, undefined_value=undefined_value, refers_to=refers_to,
                         name=name, parallel_events=parallel_events, push_down=push_down,
                         allow_learning=allow_learning)


class SeasonalEvent(Event):
    def __init__(self, event, default=0, fill_with_previous=True,
                 year:int=None, month:int=None, day:int=None,
                 hour:int=None, minute:int=None, second:int=None, microsecond:int=None,
                 is_weekday=None, is_weekend:bool=None,
                 name=None, parallel_events=None, push_down=True,
                 allow_learning=True):
        name = self._default_name_if_none(name)

        self.event = event
        self.default = self._wrapped_param(name, 'default', default)
        self.fill_with_previous = fill_with_previous
        self.year = self._wrapped_param(name, 'year', year,
                                        require_lower_bound=0.0)
        self.month = self._wrapped_param(name, 'month', month,
                                         require_lower_bound=0.0)
        self.day = self._wrapped_param(name, 'day', day,
                                       require_lower_bound=0.0)
        self.hour = self._wrapped_param(name, 'hour', hour,
                                        require_lower_bound=0.0)
        self.minute = self._wrapped_param(name, 'minute', minute,
                                          require_lower_bound=0.0)
        self.second = self._wrapped_param(name, 'second', second,
                                          require_lower_bound=0.0)
        self.microsecond = self._wrapped_param(name, 'microsecond', microsecond,
                                               require_lower_bound=0.0)

        if is_weekday is not None and is_weekday:
            self.weekday = self._wrapped_param(name, 'weekday', [0,1,2,3,4])
        else:
            self.weekday = None

        if is_weekend is not None and is_weekend:
            self.weekend = self._wrapped_param(name, 'weekend', [5, 6])
        else:
            self.weekend = None

        # what was the last value generated by the underlying event that was not blocked?
        self._last_accepted_value = None

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):

        def aux_match(a, b, cont):

            if a is not None:
                if is_sequence(a):
                    # OR semantics: any list element has the desired value?
                    for x in a:
                        if x.execute(t) == b:
                            return cont

                    # if we got here, no list element matched the desired one
                    return False

                elif a.execute(t) == b:
                    return cont

                else:
                    return False
            else:
                # if no constraint was specified, just go on
                return cont

        is_season = \
            aux_match(self.year, t.year,
                      aux_match(self.month, t.month,
                                aux_match(self.day, t.day,
                                          aux_match(self.hour, t.hour,
                                                    aux_match(self.minute, t.minute,
                                                              aux_match(self.second, t.second,
                                                                        aux_match(self.microsecond, t.microsecond,
                                                                                  aux_match(self.weekday, t.weekday(),
                                                                                            aux_match(self.weekend, t.weekday(),
                                                                                                      True)))))))))
        if is_season:
            res = self.event.execute(t)
            self._last_accepted_value = res

        else:
            if self.fill_with_previous and self._last_accepted_value is not None:
                res = self._last_accepted_value
            elif self.fill_with_previous and (self._last_accepted_value is None) and (self.default is not None):
                res = self.default.execute(t)
            elif (not self.fill_with_previous) and (self.default is not None):
                res = self.default.execute(t)
            else:
                raise ValueError("Either a base value or a previous value for the underlying event must be available,"
                                 "but both were None.")

        return res


class Choice(Event):

    def __init__(self, events, fix_choice=False, name=None, parallel_events=None, push_down=True, allow_learning=True):
        """

        :param events: When given a list of events, a uniform distribution will be assumed. When given a dict,
                       keys are events and values are their probabilities according to a categorical distribution.
                       The choice can be taken either once or at all executions.
        :param fix_choice: Whether the choice, once made, should be permanent.
        :param name:
        """
        self._fix_choice = fix_choice
        self._fixed_choice = None

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

        super().__init__(name, parallel_events, push_down, allow_learning)

    def sample_from_definition(self, t):
        raise NotImplementedError("Choice cannot be sampled directly from.")

    def sample_from_learned_distribution(self):
        raise NotImplementedError("Choice cannot be sampled directly from.")

    def choose(self, t):
        # choose
        if self._fixed_choice is not None:
            idx = self._fixed_choice
        else:
            idx = choice(range(0, len(self.events)), p=self.probs)

        # fix choice if necessary
        if self._fix_choice and self._fixed_choice is None:
            self._fixed_choice = idx

        return self.events[idx]

    def _execute(self, t, i):
        return self.choose(t).execute(t)

    def reset(self):
        super().reset()
        self._fixed_choice = None


class Piecewise(Event):

    def __init__(self, events: list, t_separators:list=None,
                 name=None, parallel_events=None, push_down=True,
                 allow_learning=True):

        self.events = events
        self.t_separators = t_separators

        self._cur_separator_pos = 0
        self._cur_separator = None

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        if self._cur_separator_pos < len(self.t_separators):
            self._cur_separator = self.t_separators[self._cur_separator_pos]

            if i >= self._value_or_execute_if_event(f'cur_separator_pos_{self._cur_separator_pos}', self._cur_separator,
                                                    t):
                self._cur_separator_pos += 1

        return self.events[self._cur_separator_pos].execute(t)

    def reset(self):
        super().reset()
        self._cur_separator_pos = 0
        self._cur_separator = None

# TODO  class RandomPiecewise(Event):
#
# def __init__(self, events: list, t_separators:list=None, spacing_mean:float=None, spacing_sd:float=None, mode: str='fixed',
#                  name=None, parallel_events=None, random_seed=None):

class TemporarySwitch(Event):
    """
    Temporarily switches results from a main event to an alternative one. Once a switch is made, it remains in force
    for a determined number of steps.
    """
    def __init__(self, main_event, alternative_event, switch_duration=1, switch_probability=0.5,
                 name=None, parallel_events=None, push_down=True,
                 allow_learning=True):

        self.main_event = self._wrapped_param(name, 'main_event', main_event)
        self.alternative_event = self._wrapped_param(name, 'alternative_event', alternative_event)
        self.switch_probability = self._wrapped_param(name, 'switch_probability', switch_probability,
                                                      require_lower_bound=0.0,
                                                      require_upper_bound=1.0)
        self.switch_duration = self._wrapped_param(name, 'switch_duration', switch_duration,
                                                   require_lower_bound=0.0)

        self.is_switched = False # always begin with main_event
        self.unswitch_step = None

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):

        if not self.is_switched:
            switch_probability_event = BernoulliEvent(self.switch_probability.execute(t))

            if switch_probability_event.execute(t) == 1:
                self.is_switched = True
                self.unswitch_step = i + self.switch_duration.execute(t)

            return self.main_event.execute(t)

        else:  # is switched
            if i >= self.unswitch_step:
                self.is_switched = False
                self.unswitch_step = None

            return self.alternative_event.execute(t)

    def reset(self):
        super().reset()
        self.is_switched = False
        self.unswitch_step = None


class Replicated(Event):
    """
    Continuously runs a clone of the underlying event for a certain specified duration, re-cloning the event
    and restarting the process at the end of each period.
    """

    def __init__(self, event, duration_per_replication, max_replication=3,
                 name=None, parallel_events=None, push_down=True, allow_learning=True):

        # WARNING: events must be pre-instantiated before use because, when learning is performed,
        #          all relevant variables must already exist, otherwise the optimizer will not have them
        #          when the causal parameters closure is calculated in the beginning of the optimization.
        self.events = [event.clone() for i in range(0, max_replication)]

        self.duration_per_replication = self._wrapped_param(name, 'duration_per_replication', duration_per_replication,
                                                            require_lower_bound=1.0)
        self.max_replication = max_replication

        self.current_event_pos = 0
        self.replicate_at_step = None

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        # Should we replicate?
        if self.replicate_at_step is None:
            self.current_event_pos = 0
            self.replicate_at_step = self.duration_per_replication.execute(t)

        elif self.replicate_at_step == i:
            self.current_event_pos = min(self.max_replication, self.current_event_pos + 1)
            self.replicate_at_step = self.replicate_at_step + self.duration_per_replication.execute(t)

        return self.events[self.current_event_pos].execute(t)

    def reset(self):
        super().reset()
        self.current_event_pos = 0
        self.replicate_at_step = None

