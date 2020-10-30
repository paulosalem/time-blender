from time_blender.core import Event, ConstantEvent
import numpy as np


class IdentityEvent(Event):
    def __init__(self, event, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)
        self.event = event

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        return self.event.execute(t)


class ClipEvent(Event):
    def __init__(self, event, max_value=None, min_value=None, name=None, parallel_events=None, push_down=True,
                 allow_learning=True):
        name = self._default_name_if_none(name)

        self.event = event
        self.max_value = self._wrapped_param(name, 'max_value', max_value)
        self.min_value = self._wrapped_param(name, 'min_value', min_value)

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        v = self.event.execute(t)

        if self.max_value is not None and v > self._value_or_execute_if_event('max_value', self.max_value, t):
            v = self.max_value.constant
        elif self.min_value is not None and v < self._value_or_execute_if_event('min_value', self.min_value, t):
            v = self.min_value.constant

        return v


class ClockEvent(Event):

    def __init__(self, as_ticks=True, name=None, parallel_events=None, push_down=True, allow_learning=True):
        name = self._default_name_if_none(name)

        self.as_ticks = as_ticks

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        if self.as_ticks:
            return i
        else:
            return t


class WaveEvent(Event):

    def __init__(self, period, amplitude, pos=0.0, name=None, parallel_events=None, push_down=True,
                 allow_learning=True):
        super().__init__(name, parallel_events, push_down, allow_learning)
        name = self._default_name_if_none(name)

        self.period = self._wrapped_param(name, 'period', period)
        self.amplitude = self._wrapped_param(name, 'amplitude', amplitude)

        self.pos = self._wrapped_param(name, 'pos', pos)
        self.ini_pos = pos

        super().__init__(name, parallel_events, push_down)

    def _execute(self, t, i):
        step = 2 * np.pi / self._value_or_execute_if_event('period', self.period, t)
        self.pos.constant += step
        self.pos.constant = self.pos.constant % (2*np.pi)
        return self._value_or_execute_if_event('amplitude', self.amplitude, t) * np.sin(self.pos.constant)

    def reset(self):
        self.pos.constant = self.ini_pos
        super().reset()


class WalkEvent(Event):

    def __init__(self, step, initial_pos=None, name=None, parallel_events=None, push_down=True, allow_learning=True,
                 capture_parent_value=True):

        self.step = self._wrapped_param(name, 'step', step)
        if initial_pos is not None:
            self.pos = self._wrapped_param(name, 'pos', initial_pos)
            if not isinstance(self.pos, ConstantEvent):
                raise ValueError("The initial position must be a constant.")

            self.ini_pos = self.pos.clone()
        else:
            self.pos = ConstantEvent(0)
            self.ini_pos = None

        self.capture_parent_value = capture_parent_value

        super().__init__(name, parallel_events, push_down, allow_learning)

    def _execute(self, t, i):
        self.pos.constant = self.pos.constant + self._value_or_execute_if_event('step', self.step, t)
        return self.pos.constant

    def _capture_push_down_value(self, t, parent_value):
        if self.capture_parent_value:
            self.pos.constant = parent_value
        # else, ignore parent value

    def reset(self):
        if self.ini_pos is not None:
            self.pos = self.ini_pos.clone()
        else:
            self.pos = ConstantEvent(0)

        super().reset()
