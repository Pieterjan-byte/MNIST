import torch
from nesy.logic import And, Or, Leaf
import copy

class Evaluator():
    """Class that represents the evaluation of the And-Or-Trees
    """

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, queries, i):
        """Evaluate all the And-Or-Trees using the specified semantics

        Args:
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            and_or_trees (List[Term]): Current Trees containing the proof for the queries
            queries (list[Term]): queries asking for the probability which images sum up to a sum
            i (int): index used to choose the correct images for the job

        Returns:
            torch.Tensor: Tensor containing probabilities
        """
        results = []
        for and_or_tree in and_or_trees:
            result = self.evaluate_tree(and_or_tree, tensor_sources, i)[0]
            results.append(result)

        return torch.stack(results)

    def evaluate_tree(self, node, tensor_sources, i):
        """Evaluate an And-Or-Tree

        Args:
            node (Term): current position in the And-Or-Tree
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            i (int): index used to choose the correct images for the job

        Returns:
            float: the evaluation of the And-Or-Tree based on the specified semantics
        """
        if isinstance(node, Leaf):
            return self.evaluate_leaf(node, tensor_sources, i)
        
        elif isinstance(node, And):
            children_values = [self.evaluate_tree(child, tensor_sources, i) for child in node.children]
            return self.label_semantics.conjunction(*children_values)
        
        elif isinstance(node, Or):
            children_values = [self.evaluate_tree(child, tensor_sources, i) for child in node.children]
            if len(children_values) == 1:
                return children_values[0]

            return self.label_semantics.disjunction(*children_values)

    def evaluate_leaf(self, leaf, tensor_sources, i):
        """Evaluate a leaf node of an And-Or-Tree

        Args:
            leaf (Term): Leaf node of the And-Or-Tree
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            i (int): index used to choose the correct images for the job

        Returns:
            float: probability
        """
        neural_predicate = leaf.term.functor
        arguments = leaf.term.arguments

        source = arguments[0]
        nb_index = arguments[1]

        tensor_name = str(source.arguments[0])
        image_index = int(source.arguments[1].functor)

        nb_index = int(arguments[1].functor)

        tensor_source = copy.deepcopy(tensor_sources)
        tensor_source['images'] = tensor_sources['images'][i:]

        prob = self.neural_predicates[neural_predicate](tensor_source[tensor_name][:, image_index])[:, nb_index]

        return prob