from abc import ABC, abstractmethod
import torch


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
    def conjunction(self, a, b):
        return a * b

    # Multiplacation of a and b: OR(a,b) = a * b
    def disjunction(self, a, b):
        return a + b - (a * b)

    # Flip the truth value or negate the value? NEG(a) = -a
    def negation(self, a):
        return 1 - a


class LukasieviczTNorm(Semantics):

    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    # T-Norm: AND(a,b) = max(0, a + b - 1)
    def conjunction(self, a, b):
        return torch.clamp(a + b - 1, min=0)

    # T-norm: OR(a,b) = min(1, a + b)
    def disjunction(self, a, b):
        return torch.clamp(a + b, max=1)

    # T-Norm: NEG(a) = max(0, 1 - a)
    def negation(self, a):
        return 1 - a

class GodelTNorm(Semantics):
    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    def conjunction(self, a, b):
        return torch.min(a, b)

    def disjunction(self, a, b):
        return torch.max(a, b)

    def negation(self, a):
        return 1 - a

class ProductTNorm(Semantics):
    # TODO: Implement this
    # We need to assign fuzzy truth values, range between 0 (completely false) and 1 (completely true)

    def conjunction(self, a, b):
        return a * b

    def disjunction(self, a, b):
        return a + b - a * b

    def negation(self, a):
        return 1 - a
