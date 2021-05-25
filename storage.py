"""
    Provides some methods for optimising calculations of
"""


from .helpers import format_label
import functools
import pickle


def determine_symmetries(symbol, *args):
    """
        Determines symmetries of Wigner symbols
    """
    if symbol == '3j':

        # Unpack arguments
        j1, j2, j, m1, m2, m = args

        # Define permutations where the 3j symbol is equal
        equal = [
            format_label(j2, j, j1, m2, m, m1),
            format_label(j, j1, j2, m, m1, m1)
        ]

        # Define permutations where the 3j symbol gains a prefactor (-1)^(j1+j2+j)
        prefactor = [
            format_label(j2, j1, j, m2, m1, m),
            format_label(j1, j, j2, m1, m, m2),
            format_label(j, j2, j1, m, m2, m1),
            format_label(j1, j2, j, -1*m1, -1*m2, -1*m)
        ]

        return {1: equal, (-1)**(j1+j2+j): prefactor}
    elif symbol == '6j':

        # Unpack arguments
        j1, j2, j3, J1, J2, J3 = args

        # Define permutations where the 6j symbol is equal
        equal = [
            format_label(j2, j1, j3, J2, J1, J3),
            format_label(j3, j1, j2, J3, J1, J2),
            format_label(J1, J2, j3, j1, j2, J3),
            format_label(J1, j2, J3, j1, J2, j3),
            format_label(j1, J2, J3, J1, j2, j3)
        ]

        return {1: equal}


def precalc(symbol):
    """
        A decorator which checks whether a function has been evaluated previously for the given arguments
        and, if it has, loads the value from a pickle file. If it hasn't, it runs the function and saves the value
        for future runs.
    """
    def precalc_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            symbols = {'3j': {}, '6j': {}}
            try:
                with open('symbol_storage.pickle', 'rb') as f:
                    symbols = pickle.load(f)
            except:
                pass

            label = format_label(*args)
            if label in symbols[symbol]:
                return symbols[symbol][label]

            value = func(*args, **kwargs)
            symmetries = determine_symmetries(symbol, *args)
            symbols[symbol][label] = value
            for prefactor in symmetries.keys():
                for sym_label in symmetries[prefactor]:
                    symbols[symbol][sym_label] = prefactor*value
            with open('symbol_storage.pickle', 'wb+') as f:
                pickle.dump(symbols, f)

            return value
        return wrapper
    return precalc_decorator
