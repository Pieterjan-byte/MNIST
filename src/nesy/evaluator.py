import torch
from nesy.logic import And, Or, Leaf

class Evaluator():
    """
    The Evaluator class is responsible for evaluating And-Or trees, and to provide a final score for each query

    It traverses the tree, assigning values to each node based on whether it's a leaf with a neural predicate
    or a logical operator (AND/OR). The evaluation uses specified semantics for continuous operations, suitable for probabilistic or fuzzy logic

    Attributes:
        label_semantics (Semantics): The semantics used for evaluating logical expressions in the tree
        neural_predicates (torch.nn.ModuleDict): A dictionary mapping neural predicate names to their corresponding neural network modules

    """

    # Corresponding to our implementation of the logic class
    # this implementation assumes that all leaf nodes are associated with neural predicates and that the constructed And-Or trees
    # do not contain known facts as leaf nodes, hence no leaf node is directly assigned a value 1.


    def __init__(self, label_semantics, neural_predicates):
        """
        Initializes the Evaluator with given label semantics and neural predicates

        Args:
            label_semantics (Semantics): The semantics for logical operations evaluation
            neural_predicates (torch.nn.ModuleDict): Neural predicates mapped to neural network modules
        """
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, i):
        """
        Evaluate all the And-Or-Trees using the specified semantics

        Args:
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            and_or_trees (List[Term]): Current Trees containing the proof for the current list of queries
            i (int): index used to choose the correct images for the job

        Returns:
            torch.Tensor: A tensor containing the probabilities for the queries
        """
        results = []
        #print("And or trees:", and_or_trees)
        for and_or_tree in and_or_trees: # Iterate over the different and-or-trees, each one corresponding to a query in a query group
            result = self.evaluate_tree(and_or_tree, tensor_sources, i)[0]
            results.append(result)

        return torch.stack(results)

    def evaluate_tree(self, node, tensor_sources, i):
        """
        Recursively evaluates an And-Or tree from a given node

        Args:
            node (Term): current position(=node) in the And-Or-Tree being evaluated
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            i (int): index used to choose the correct images for the job

        Returns:
            float: the evaluation of the subtree rooted at 'node' based on the specified semantics
        """
        if isinstance(node, Leaf):
            return self.evaluate_leaf(node, tensor_sources, i)
        else:
            # Evaluate all child nodes and apply conjunction/disjunction semantics
            children_values = [self.evaluate_tree(child, tensor_sources, i) for child in node.children]
            operation = self.label_semantics.conjunction if isinstance(node, And) else self.label_semantics.disjunction
            return operation(*children_values)


    def evaluate_leaf(self, leaf, tensor_sources, i):
        """
        Evaluate a leaf node of an And-Or-Tree: assign the corresponding neural network output

        Args:
            leaf (Term): Leaf node of the And-Or-Tree, representing a neural predicate
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            i (int): index used to choose the correct images out of tensor_sources

        Returns:
            float: probability associated with the neural predicate
        """

        # Extract the neural predicate and its arguments
        neural_predicate = leaf.term.functor # normally this equals 'digit'
        arguments = leaf.term.arguments

        source = arguments[0]
        nb_index = int(arguments[1].functor)

        # Extract tensor information and index for evaluation
        tensor_name = str(source.arguments[0]) # normally this equals 'images'
        image_index = int(source.arguments[1].functor)

        # Access the required subset of images
        selected_images = tensor_sources[tensor_name][i:]

        # Invoke the neural predicate with selected tensor and return the result
        prob = self.neural_predicates[neural_predicate](selected_images[:, image_index])[:, nb_index]

        return prob