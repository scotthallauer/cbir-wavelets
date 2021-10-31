# https://realpython.com/python-timer/

import time

class TimerError(Exception):
  """A custom exception used to report errors in use of Timer class"""

class Timer:
  def __init__(self):
    self._start_time = None
    self._elapsed_time = None

  def start(self):
    """Start a new timer"""
    if self._start_time is not None:
      raise TimerError(f"Timer is running. Use .stop() to stop it")
    self._start_time = time.perf_counter()
    self._elapsed_time = None

  def stop(self):
    """Stop the timer"""
    if self._start_time is None:
      raise TimerError(f"Timer is not running. Use .start() to start it")
    self._elapsed_time = time.perf_counter() - self._start_time
    self._start_time = None

  def time(self):
    """Return the elapsed time of the last completed timer"""
    if self._elapsed_time is None and self._start_time is not None:
      raise TimerError(f"Timer is running. Use .stop() to stop it")
    elif self._elapsed_time is None and self._start_time is None:
      raise TimerError(f"No timer has run yet. Use .start() to start one")
    return self._elapsed_time