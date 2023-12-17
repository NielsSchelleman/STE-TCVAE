import numpy as np
from itertools import chain, combinations


class AbsoluteWRAcc:

    def __init__(self, total_size, total_acc, exponent_func=np.sqrt):
        """
        Computes the WRACC quality measure with some arbirary exponentiating function
        :param total_size: total size of the target
        :param total_acc:  total accuracy of the target
        :param exponent_func: scaling function
            Examples:
            classic WRAcc: lambda x: x
            biased towards smaller subgroups: np.sqrt
            biased towards larger subgroups: np.square
            larger bias towards smaller subgroups: lambda x: x**0.2
        """
        self.total_size = total_size
        self.total_acc = total_acc

        self.exponent_func = exponent_func

        max_score = min(self.total_acc, 1-self.total_acc)
        self.scaling_factor = self.exponent_func(max_score)*(1-max_score)

    def compute(self, size, acc) -> float:
        return self.exponent_func(size/self.total_size)*abs(acc-self.total_acc)

    def compute_scaled(self, size, acc) -> float:
        return self.compute(size, acc) / self.scaling_factor


def powerset(iterable):
    """
    Taken From:
    https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

