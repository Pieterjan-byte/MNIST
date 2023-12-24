import torch
from nesy.logic import And, Or, Leaf

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, queries):
        # TODO: Implement this
        results = []
        for and_or_tree in and_or_trees:
            result = self.evaluate_tree(and_or_tree, tensor_sources)
            results.append(result)
        return torch.stack(results)

    def evaluate_tree(self, node, tensor_sources):
        if isinstance(node, Leaf):
            return self.evaluate_leaf(node, tensor_sources)
        elif isinstance(node, And):
            children_values = [self.evaluate_tree(child, tensor_sources) for child in node.children]
            return self.label_semantics.conjunction(*children_values)
        elif isinstance(node, Or):
            children_values = [self.evaluate_tree(child, tensor_sources) for child in node.children]
            return self.label_semantics.disjunction(*children_values)

    def evaluate_leaf(self, leaf, tensor_sources):
        # Assuming leaf is a neural predicate
        neural_predicate = leaf.term.functor
        arguments = leaf.term.arguments
        # The neural predicate function is called with the corresponding tensors
        return self.neural_predicates[neural_predicate](*[tensor_sources[arg] for arg in arguments])

        # Step 1: Traverse the And-Or tree
        # (Implement logic to traverse the tree and calculate probabilities at each node)


        # Our dummy And-Or-Tree (addition(img0, img1,0) is represented by digit(img0,0) AND digit(img1,0)
        # The evaluation is:
        # p(addition(img0, img1,0)) = p(digit(img0,0) AND digit(img1,0)) =
        """p_digit_0_0 = self.neural_predicates["digit"](tensor_sources["images"][:,0])[:,0]
        p_digit_1_0 = self.neural_predicates["digit"](tensor_sources["images"][:,1])[:,0]
        p_sum_0 =  p_digit_0_0 * p_digit_1_0

        # Here we trivially return the same value (p_sum_0[0]) for each of the queries to make the code runnable
        if isinstance(queries[0], list):
            res = [torch.stack([p_sum_0[0] for q in query]) for query in queries]
        else:
            res = [p_sum_0[0] for query in queries]
        return torch.stack(res)"""