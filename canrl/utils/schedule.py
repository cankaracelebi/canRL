"""Schedules for learning rate, epsilon, and other annealing."""

from abc import ABC, abstractmethod


class Schedule(ABC):
    """Base class for parameter schedules."""
    
    @abstractmethod
    def value(self, step: int) -> float:
        """Get value at given step."""
        pass
    
    def __call__(self, step: int) -> float:
        return self.value(step)


class LinearSchedule(Schedule):
    """
    Linear interpolation between start and end values.
    
    Args:
        start: Initial value.
        end: Final value.
        duration: Number of steps to reach end value.
        
    Example:
        >>> # Epsilon decay from 1.0 to 0.01 over 10000 steps
        >>> schedule = LinearSchedule(1.0, 0.01, 10000)
        >>> schedule(0)     # 1.0
        >>> schedule(5000)  # 0.505
        >>> schedule(10000) # 0.01
        >>> schedule(20000) # 0.01 (stays at end)
    """
    
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = duration
    
    def value(self, step: int) -> float:
        """Get linearly interpolated value."""
        if step >= self.duration:
            return self.end
        
        progress = step / self.duration
        return self.start + (self.end - self.start) * progress


class ExponentialSchedule(Schedule):
    """
    Exponential decay from start to end value.
    
    Args:
        start: Initial value.
        end: Final (asymptotic) value.
        decay_rate: Decay rate per step.
        
    Example:
        >>> schedule = ExponentialSchedule(1.0, 0.01, 0.9999)
        >>> schedule(0)     # 1.0
        >>> schedule(1000)  # ~0.095
    """
    
    def __init__(self, start: float, end: float, decay_rate: float):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate
    
    def value(self, step: int) -> float:
        """Get exponentially decayed value."""
        decayed = self.start * (self.decay_rate ** step)
        return max(self.end, decayed)


class ConstantSchedule(Schedule):
    """Constant value schedule."""
    
    def __init__(self, value: float):
        self._value = value
    
    def value(self, step: int) -> float:
        return self._value


class PiecewiseSchedule(Schedule):
    """
    Piecewise linear schedule with multiple phases.
    
    Args:
        endpoints: List of (step, value) tuples defining the schedule.
        
    Example:
        >>> schedule = PiecewiseSchedule([
        ...     (0, 1.0),
        ...     (1000, 0.5),
        ...     (5000, 0.1),
        ...     (10000, 0.01),
        ... ])
    """
    
    def __init__(self, endpoints: list[tuple[int, float]]):
        self.endpoints = sorted(endpoints, key=lambda x: x[0])
    
    def value(self, step: int) -> float:
        """Get piecewise interpolated value."""
        # Before first endpoint
        if step <= self.endpoints[0][0]:
            return self.endpoints[0][1]
        
        # After last endpoint
        if step >= self.endpoints[-1][0]:
            return self.endpoints[-1][1]
        
        # Find surrounding endpoints
        for i in range(len(self.endpoints) - 1):
            start_step, start_val = self.endpoints[i]
            end_step, end_val = self.endpoints[i + 1]
            
            if start_step <= step < end_step:
                progress = (step - start_step) / (end_step - start_step)
                return start_val + (end_val - start_val) * progress
        
        return self.endpoints[-1][1]
