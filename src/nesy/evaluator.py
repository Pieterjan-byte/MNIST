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
            result = self.evaluate_tree(and_or_tree, tensor_sources)[0]
            #print("\n\nresult: \n\n", result)
            results.append(result)
        return torch.stack(results)

    def evaluate_tree(self, node, tensor_sources):
        if isinstance(node, Leaf):
            #print("\n\nresult Leaf: \n\n", self.evaluate_leaf(node, tensor_sources))
            return self.evaluate_leaf(node, tensor_sources)
        elif isinstance(node, And):
            children_values = [self.evaluate_tree(child, tensor_sources) for child in node.children]
            #print("\n\nresult AND: \n\n", self.label_semantics.conjunction(*children_values))
            return self.label_semantics.conjunction(*children_values)
        elif isinstance(node, Or):
            children_values = [self.evaluate_tree(child, tensor_sources) for child in node.children]
            if len(children_values) == 1:
                #print("\n\nresult FINAL OR: \n\n", children_values[0] )
                return children_values[0]  # Return the single child's value directly
            #print("\n\nresult OR: \n\n", self.label_semantics.disjunction(*children_values))
            #print(children_values)
            return self.label_semantics.disjunction(*children_values)

    def evaluate_leaf(self, leaf, tensor_sources):

        # Assuming leaf is a neural predicate
        neural_predicate = leaf.term.functor
        arguments = leaf.term.arguments

        source = arguments[0]
        nb_index = arguments[1]

        #print("\n\n Neural predicate :\n  ", neural_predicate, "\n  arguments:\n  ",  arguments)
        #print("\n\n source:\n  ", source, "\n  nb_index:\n  ",  nb_index)
        #print("\n\nAvailable Keys in tensor_sources:\n\n", tensor_sources.keys())

        # Parse the argument assuming the format 'tensor(images,0)'
        tensor_name = str(source.arguments[0])
        image_index = int(source.arguments[1].functor)

        # Assuming the second argument of the leaf is an index for the neural predicate output
        nb_index = int(arguments[1].functor)

        #print("\n  tensor name:\n  ",  tensor_name, "\n  image_index:\n  ",  image_index, "\n \n ")

        prob = self.neural_predicates[neural_predicate](tensor_sources[tensor_name][:, image_index])[:, nb_index]

        #print("\n\nprob: \n\n", prob)

        # The neural predicate function is called with the corresponding tensors
        return prob


        """

        def evaluate(self, tensor_sources, and_or_tree, queries):
        # TODO: Implement this


        # Our dummy And-Or-Tree (addition(img0, img1,0) is represented by digit(img0,0) AND digit(img1,0)
        # The evaluation is:
        # p(addition(img0, img1,0)) = p(digit(img0,0) AND digit(img1,0)) =
        p_digit_0_0 = self.neural_predicates["digit"](tensor_sources["images"][:,0])[:,0]
        p_digit_1_0 = self.neural_predicates["digit"](tensor_sources["images"][:,1])[:,0]
        p_sum_0 =  p_digit_0_0 * p_digit_1_0

        # Here we trivially return the same value (p_sum_0[0]) for each of the queries to make the code runnable
        if isinstance(queries[0], list):
            res = [torch.stack([p_sum_0[0] for q in query]) for query in queries]
        else:
            res = [p_sum_0[0] for query in queries]
        return torch.stack(res)


        """