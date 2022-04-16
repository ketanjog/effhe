'''
A utility class to help measure the time of experiments
'''

from time import time

class Timer():
    def __init__(self) -> None:
        pass

    def time(func):
        start = time()
        result = func()
        end = time()

        return result, (end - start)

