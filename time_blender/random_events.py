import numpy as np

from time_blender.core import Event
from scipy.stats import norm, poisson, uniform, bernoulli

from time_blender.deterministic_events import ClipEvent


class UniformEvent(Event):

    def __init__(self, low=-1.0, high=1.0, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)
        self.low = self._wrapped_param(name, 'low', low)
        self.high = self._wrapped_param(name, 'high', high)

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        l = self._value_or_execute_if_event('low', self.low, t)
        h = self._value_or_execute_if_event('high', self.high, t)

        return uniform.rvs(loc=l, scale=max(0, h-l))


class BernoulliEvent(Event):

    def __init__(self, p=0.5, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)
        self.p = self._wrapped_param(name, 'p', p, require_lower_bound=0.0, require_upper_bound=1.0)

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        p = self._value_or_execute_if_event('p', self.p, t)
        return bernoulli.rvs(p=p)


class NormalEvent(Event):

    def __init__(self, mean=0.0, std=1.0, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)
        self.mean = self._wrapped_param(name, 'mean', mean)
        self.std = self._wrapped_param(name, 'std', std, require_lower_bound=0.0)

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        loc = self._value_or_execute_if_event('mean', self.mean, t)
        scale = self._value_or_execute_if_event('std', self.std, t)

        v = norm.rvs(loc=loc, scale=scale)
        return v


class PoissonEvent(Event):

    def __init__(self, lamb=1, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)
        self.lamb = self._wrapped_param(name, 'lamb', lamb, require_lower_bound=0.0)

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        l = self._value_or_execute_if_event('lamb', self.lamb, t)
        return poisson.rvs(mu=l)


class Resistance(Event):

    def __init__(self, event, resistance_value_begin, resistance_value_end,
                 resistance_probability, resistance_strength_event, direction, name=None, parallel_events=None,
                 push_down=True, allow_learning=True):

        self.event = event
        self.resistance_value_begin = self._wrapped_param(name, 'resistance_value_begin', resistance_value_begin)
        self.resistance_value_end = self._wrapped_param(name, 'resistance_value_end', resistance_value_end)
        self.resistance_probability = self._wrapped_param(name, 'resistance_probability',
                                                          resistance_probability,
                                                          require_lower_bound=0.0, require_upper_bound=1.0)
        self.resistance_strength_event = self._wrapped_param(name, 'resistance_strength_event',
                                                             resistance_strength_event,
                                                             require_lower_bound=0.0, require_upper_bound=1.0)
        self.direction = direction

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        value = self.event.execute(t)

        # Decide whether to resist
        rand_top = uniform.rvs()  # [0, 1]
        rand_bottom = uniform.rvs()  # [0, 1]
        if self.direction == 'top' and \
           self.resistance_value_begin.constant <= value < self.resistance_value_end.constant and \
           self.resistance_probability.constant > rand_top:

                resist = True

        elif self.direction == 'bottom' and \
             self.resistance_value_end.constant < value <= self.resistance_value_begin.constant and \
             self.resistance_probability.constant > rand_bottom:

                resist = True
        else:
            resist = False

        # apply resistance, if pertinent
        if resist:
            resistance_factor = self.resistance_strength_event.execute(t)
            assert 0.0 <= resistance_factor, f"Resistance factor must be >= 0.0, but was {resistance_factor}."

            if self.direction == 'top':
                # pushes downward
                return value - abs(value) * resistance_factor

            else:
                # pushes upward
                return value + abs(value) * resistance_factor
        else:
            # no resistance applied
            return value


class TopResistance(Resistance):
    def __init__(self, event, resistance_value_begin, resistance_value_end, resistance_probability,
                 resistance_strength_event, name=None, parallel_events=None, push_down=True, allow_learning=True):
        super().__init__(event=event,
                         resistance_value_begin=resistance_value_begin,
                         resistance_value_end=resistance_value_end,
                         resistance_probability=resistance_probability,
                         resistance_strength_event=resistance_strength_event,
                         direction='top',
                         name=name, parallel_events=parallel_events, push_down=push_down,
                         allow_learning=allow_learning)


class BottomResistance(Resistance):
    def __init__(self, event, resistance_value_begin, resistance_value_end, resistance_probability,
                 resistance_strength_event, name=None, parallel_events=None, push_down=True, allow_learning=True):
        super().__init__(event=event,
                         resistance_value_begin=resistance_value_begin,
                         resistance_value_end=resistance_value_end,
                         resistance_probability=resistance_probability,
                         resistance_strength_event=resistance_strength_event,
                         direction='bottom',
                         name=name, parallel_events=parallel_events, push_down=push_down,
                         allow_learning=allow_learning)


def wrap_in_resistance(event, top_resistance_levels=[], bottom_resistance_levels=[],
                       top_resistance_strength_event=ClipEvent(NormalEvent(0.02, 0.01), min_value=0.0),
                       bottom_resistance_strength_event=ClipEvent(NormalEvent(0.02, 0.01), min_value=0.0),
                       tolerance=5, top_resistance_probability=0.5, bottom_resistance_probability=0.5):

    if len(top_resistance_levels) > 0:
        level = top_resistance_levels[0]
        res = TopResistance(event,
                            resistance_value_begin=level - tolerance,
                            resistance_value_end=level,
                            resistance_probability=top_resistance_probability,
                            resistance_strength_event=top_resistance_strength_event)
        # recursive step
        res = wrap_in_resistance(res,
                                 top_resistance_levels=top_resistance_levels[1:],
                                 bottom_resistance_levels=bottom_resistance_levels,
                                 top_resistance_strength_event=top_resistance_strength_event,
                                 bottom_resistance_strength_event=bottom_resistance_strength_event,
                                 tolerance=tolerance,
                                 top_resistance_probability=top_resistance_probability,
                                 bottom_resistance_probability=bottom_resistance_probability)

    elif len(bottom_resistance_levels) > 0:
        level = bottom_resistance_levels[0]
        res = BottomResistance(event,
                               resistance_value_begin=level + tolerance,
                               resistance_value_end=level,
                               resistance_probability=bottom_resistance_probability,
                               resistance_strength_event=bottom_resistance_strength_event)

        # recursive step
        res = wrap_in_resistance(res,
                                 top_resistance_levels=top_resistance_levels,
                                 bottom_resistance_levels=bottom_resistance_levels[1:],
                                 top_resistance_strength_event=top_resistance_strength_event,
                                 bottom_resistance_strength_event=bottom_resistance_strength_event,
                                 tolerance=tolerance,
                                 top_resistance_probability=top_resistance_probability,
                                 bottom_resistance_probability=bottom_resistance_probability)
    else:
        # recursion base
        res = event

    return res

# TODO bubble

# TODO mean-reverting models
