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
    def conjunction(self, a, b):
        return a * b

    def disjunction(self, a, b):
        return a + b - (a * b)

    def negation(self, a):
        return 1 - a


class LukasieviczTNorm(Semantics):
    def conjunction(self, a, b):
        return torch.clamp(a + b - 1, min=0)

    def disjunction(self, a, b):
        return torch.clamp(a + b, max=1)

    def negation(self, a):
        return 1 - a


class GodelTNorm(Semantics):
    def conjunction(self, a, b):
        return torch.min(a, b)

    def disjunction(self, a, b):
        return torch.max(a, b)

    def negation(self, a):
        return 1 - a


class ProductTNorm(Semantics):
    def conjunction(self, a, b):
        return a * b

    def disjunction(self, a, b):
        return a + b - a * b

    def negation(self, a):
        return 1 - a
