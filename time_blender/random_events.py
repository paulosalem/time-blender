from time_blender.core import RandomEvent
import numpy as np
import pymc3 as pm


class UniformEvent(RandomEvent):

    def __init__(self, low=-1.0, high=1.0, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.low = low
        self.high = high

        super().__init__(lambda model, obs: pm.Uniform(self.name,
                                                       lower=self._pymc3_model_variables_if_distribution(model, low),
                                                       upper=self._pymc3_model_variables_if_distribution(model, high),
                                                       observed=obs),
                         name, parallel_events, push_down)

    def sample_from_definition(self, t):
        return np.random.uniform(self._value_or_execute_if_event(self.low, t),
                                 self._value_or_execute_if_event(self.high, t))


class NormalEvent(RandomEvent):

    def __init__(self, mean, std, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.mean = mean
        self.std = std

        super().__init__(lambda model, obs: pm.Normal(self.name,
                                                      mu=self._pymc3_model_variables_if_distribution(model, mean),
                                                      sd=self._pymc3_model_variables_if_distribution(model, std),
                                                      observed=obs),
                         name, parallel_events, push_down)

    def sample_from_definition(self, t):
        return np.random.normal(self._value_or_execute_if_event(self.mean, t),
                                self._value_or_execute_if_event(self.std, t))


class PoissonEvent(RandomEvent):

    def __init__(self, lamb, name=None, parallel_events=None, push_down=False):
        name = self._default_name_if_none(name)
        self.lamb = lamb

        super().__init__(lambda model, obs: pm.Poisson(self.name,
                                                       mu=self._pymc3_model_variables_if_distribution(model, lamb),
                                                       observed=obs),
                         name, parallel_events, push_down)

    def sample_from_definition(self, t):
        return np.random.poisson(self._value_or_execute_if_event(self.lamb, t))


class Resistance(RandomEvent):

    def __init__(self, event, resistance_value_begin, resistance_value_end,
                 resistance_probability, resistance_strength_event, direction, name=None, parallel_events=None,
                 push_down=False):

        self.event = event
        self.resistance_value_begin = resistance_value_begin
        self.resistance_value_end = resistance_value_end
        self.resistance_probability = resistance_probability
        self.resistance_strength_event = resistance_strength_event
        self.direction = direction

        def aux(model, obs):
            raise NotImplementedError("Not supported.")

        super().__init__(aux,
                         name, parallel_events, push_down)

    def sample_from_definition(self, t):
        value = self.event.execute(t)

        # Decide whether to resist
        if self.direction == 'top' and \
           self.resistance_value_begin <= value < self.resistance_value_end and \
           self.resistance_probability > np.random.random():

                resist = True

        elif self.direction == 'bottom' and \
             self.resistance_value_end < value <= self.resistance_value_begin and \
             self.resistance_probability > np.random.random():

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
