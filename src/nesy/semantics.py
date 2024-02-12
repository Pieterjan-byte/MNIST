from abc import ABC, abstractmethod
import math


class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass


class SumProductSemiring(Semantics):
    # TODO: Implement this

    # Summation of a and b: AND(a,b) = a + b
    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        return sum(args)

    # Flip the truth value or negate the value? NEG(a) = -a
    def negation(self, a):
        return 1 - a



class LukasieviczTNorm(Semantics):

    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    def conjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = max(0, result + arg - 1)
        return result

    def disjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = min(1, result + arg)
        return result

    # T-Norm: NEG(a) = max(0, 1 - a)
    def negation(self, a):
        return 1 - a

class GodelTNorm(Semantics):
    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    def conjunction(self, *args):
        return min(args)

    def disjunction(self, *args):
        return max(args)

    def negation(self, a):
        return 1 - a

class ProductTNorm(Semantics):
    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = result + arg - result * arg
        return result

    def negation(self, a):
        return 1 - a
