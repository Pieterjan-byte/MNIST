from abc import ABC, abstractmethod
import math
import torch


# Base class for defining semantics of logical operations. It serves as an interface for different logical systems.
class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        """Abstract method for logical conjunction (AND operation)."""
        pass

    @abstractmethod
    def disjunction(self, a, b):
        """Abstract method for logical disjunction (OR operation)."""
        pass

    @abstractmethod
    def negation(self, a):
        """Abstract method for logical negation (NOT operation)."""
        pass

# Source for formulas: “From statistical relational to neural symbolic artificial intelligence: a survey” from Giuseppe Marra et al. Available at: https://arxiv.org/pdf/2108.11451.pdf
# The conjunction and negation methods in our classes are changed to handle multiple arguments

class SumProductSemiring(Semantics):

    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        return sum(args)

    def negation(self, a):
        return 1 - a



class LukasieviczTNorm(Semantics):

    def conjunction(self, *args):
        #print("ARGS\n\n", args)
        result = torch.max(torch.zeros_like(args[0]), sum(args) - (len(args) - 1))
        return result

    def disjunction(self, *args):
        result = torch.min(torch.ones_like(args[0]), sum(args))
        return result

    def negation(self, a):
        return 1 - a

class GodelTNorm(Semantics):

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

    def conjunction(self, *args):
        return math.prod(args)

    def disjunction(self, *args):
        result = args[0]
        for arg in args[1:]:
            result = result + arg - result * arg
        return result

    def negation(self, a):
        return 1 - a