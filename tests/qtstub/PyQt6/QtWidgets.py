from PyQt6._base import *  # noqa
from PyQt6._base import Dummy as _D

def __getattr__(name):
    return type(name, (_D,), {})
