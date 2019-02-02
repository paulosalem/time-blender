from time_blender.core import Event
import numpy as np


class IdentityEvent(Event):
    def __init__(self, event, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.event = event

    def _execute(self, t, i):
        return self.event.execute(t)


class ClipEvent(Event):
    def __init__(self, event, max_value=None, min_value=None, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.event = event
        self.max_value = max_value
        self.min_value = min_value

    def _execute(self, t, i):
        v = self.event.execute(t)

        if self.max_value is not None and v > self.max_value:
            v = self.max_value
        elif self.min_value is not None and v < self.min_value:
            v = self.min_value

        return v

class ClockEvent(Event):

    def __init__(self, as_ticks=True, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.as_ticks = as_ticks

    def _execute(self, t, i):
        if self.as_ticks:
            return i
        else:
            return t


class WaveEvent(Event):

    def __init__(self, period, amplitude, pos=0.0, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.period = period
        self.amplitude = amplitude

        self.pos = pos
        self.ini_pos = pos

    def _execute(self, t, i):
        step = 2*np.pi / self._value_or_execute_if_event(self.period, t)
        self.pos += step
        self.pos = self.pos % (2*np.pi)
        return self._value_or_execute_if_event(self.amplitude, t) * np.sin(self.pos)

    def reset(self):
        self.pos = self.ini_pos
        super().reset()


class ConstantEvent(Event):
    def __init__(self, constant, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.constant = constant

    def _execute(self, t, i):
        return self._value_or_execute_if_event(self.constant, t)


class WalkEvent(Event):

    def __init__(self, pos, step, name=None, parallel_events=None, push_down=False):
        super().__init__(name, parallel_events, push_down)
        self.step = step

        self.ini_pos = pos
        self.pos = pos

    def _execute(self, t, i):
        self.pos = self.pos + self._value_or_execute_if_event(self.step, t)
        return self.pos

    def _capture_push_down_value(self, t, parent_value):
        self.pos = parent_value

    def reset(self):
        self.pos = self.ini_pos
        super().reset()
