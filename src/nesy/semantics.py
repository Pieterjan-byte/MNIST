from abc import ABC, abstractmethod
import math
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
    # TODO: Fix this because possibly not correct?

    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        return sum(args)

    def negation(self, a):
        return 1 - a



class LukasieviczTNorm(Semantics):

    # TODO: Correct?

    def conjunction(self, *args):
        print("ARGS\n\n", args)
        result = torch.max(torch.zeros_like(args[0]), sum(args) - (len(args) - 1))
        return result

    def disjunction(self, *args):
        result = torch.min(torch.zeros_like(args[0]) + 1, sum(args))
        return result

    def negation(self, a):
        return 1 - a

class GodelTNorm(Semantics):
    # TODO: Correct?

    def conjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = torch.min(result, arg)
        return result

    def disjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = torch.max(result, arg)
        return result

    def negation(self, a):
        return 1 - a

class ProductTNorm(Semantics):
    # TODO: disjunction is wrong?

    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = result + arg - result * arg
        return result

    def negation(self, a):
        return 1 - a