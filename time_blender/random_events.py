import numpy as np

from time_blender.core import Event
from scipy.stats import norm, poisson, uniform

class UniformEvent(Event):

    def __init__(self, low=-1.0, high=1.0, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.low = self._wrapped_param(name, 'low', low)
        self.high = self._wrapped_param(name, 'high', high)

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i, obs=None):
        l = self._value_or_execute_if_event('low', self.low, t)
        h = self._value_or_execute_if_event('high', self.high, t)

        return uniform.rvs(loc=l, scale=max(0, h-l))


class NormalEvent(Event):

    def __init__(self, mean, std, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.mean = self._wrapped_param(name, 'mean', mean)
        self.std = self._wrapped_param(name, 'std', std, require_lower_bound=0.0)

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i, obs=None):
        loc = self._value_or_execute_if_event('mean', self.mean, t)
        scale = self._value_or_execute_if_event('std', self.std, t)

        v = norm.rvs(loc=loc, scale=scale)
        return v


class PoissonEvent(Event):

    def __init__(self, lamb, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.lamb = self._wrapped_param(name, 'lamb', lamb, require_lower_bound=0.0)

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i, obs=None):
        l = self._value_or_execute_if_event('lamb', self.lamb, t)
        return poisson.rvs(mu=l)


class Resistance(Event):

    def __init__(self, event, resistance_value_begin, resistance_value_end,
                 resistance_probability, resistance_strength_event, direction, name=None, parallel_events=None,
                 push_down=False):
        name = self._default_name_if_none(name)

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

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i, obs=None):
        value = self.event.execute(t)

        # Decide whether to resist
        rand_top = uniform.rvs() # [0, 1]
        rand_bottom = uniform.rvs() # [0, 1]
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
            assert(0.0 <= resistance_factor <= 1.0)

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
                 resistance_strength_event, name=None, parallel_events=None, push_down=True):
        super().__init__(event=event,
                         resistance_value_begin=resistance_value_begin,
                         resistance_value_end=resistance_value_end,
                         resistance_probability=resistance_probability,
                         resistance_strength_event=resistance_strength_event,
                         direction='top',
                         name=name, parallel_events=parallel_events, push_down=push_down)


class BottomResistance(Resistance):
    def __init__(self, event, resistance_value_begin, resistance_value_end, resistance_probability,
                 resistance_strength_event, name=None, parallel_events=None, push_down=True):
        super().__init__(event=event,
                         resistance_value_begin=resistance_value_begin,
                         resistance_value_end=resistance_value_end,
                         resistance_probability=resistance_probability,
                         resistance_strength_event=resistance_strength_event,
                         direction='bottom',
                         name=name, parallel_events=parallel_events, push_down=push_down)


# TODO bubble

# TODO mean-reverting models
