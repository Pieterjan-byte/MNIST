from nesy.term import Term, Clause, Fact, Variable
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):

    def reason(self, program: tuple[Clause], queries: list[Term]):

        print("\nProgram: \n", program)
        print("\n\nQueries: \n", queries)

        # Initialize known facts
        known_facts = set()

        # Update to handle Fact objects
        for item in program:
            if isinstance(item, Fact):
                # Treat Fact as a Clause with an empty body
                known_facts.add(item.term)

        #print("\n\nKnown facts: \n", known_facts)

        # Initialize And-Or trees with placeholders
        and_or_trees = [None] * len(queries)

        # Apply forward chaining
        inferred = True
        while inferred:
            inferred = False
            # Inside the while loop for forward chaining
            for clause in program:
                if not isinstance(clause, Fact):
                    new_facts_added, and_or_trees = add_substitutions(clause, known_facts, queries, and_or_trees)
                    if new_facts_added:
                        inferred = True


        #print("\n\nKnown facts after forward chaining: \n", known_facts)
        #print("\n\nAnd-Or Trees: \n", and_or_trees, "\n\n")

        return and_or_trees

def add_substitutions(clause, known_facts, queries, and_or_trees):
    # Start with a list containing an empty substitution
    complete_substitutions = [{}]
    body = clause.body
    new_facts_added = False
    added_nodes = set()  # Set to track added node representations

    #print("\n\nClause: \n\n", clause)

    for term in body:
        new_substitutions = []
        for substitution in complete_substitutions:
            for fact in known_facts:
                if term.functor == fact.functor and len(term.arguments) == len(fact.arguments):
                    new_substitution = substitution.copy()
                    valid_substitution = True
                    for term_arg, fact_arg in zip(term.arguments, fact.arguments):
                        if isinstance(term_arg, Variable):
                            # If the variable is already in the substitution, it must match the current fact
                            if term_arg.name in new_substitution and new_substitution[term_arg.name] != fact_arg:
                                valid_substitution = False
                                break
                            new_substitution[term_arg.name] = fact_arg
                        elif term_arg != fact_arg:
                            valid_substitution = False
                            break
                    if valid_substitution:
                        new_substitutions.append(new_substitution)
        complete_substitutions = new_substitutions

    # Filter out incomplete substitutions
    complete_substitutions = [sub for sub in complete_substitutions if is_complete_substitution(body, sub)]

    #print("\n\nComplete substitutions :\n\n", complete_substitutions)

    for substitution in complete_substitutions:
        substituted_head = substitute(clause.head, substitution)
        if substituted_head not in known_facts:
            known_facts.add(substituted_head)
            new_facts_added = True
        # Check if the substituted head matches any query
        for i, query in enumerate(queries):
            if substituted_head == query:
                # Construct And-Or tree for this query
                and_node = construct_and_or_tree_node(clause, substitution)
                node_repr = repr(and_node)  # Get the string representation of the node
                if and_or_trees[i] is None:
                    and_or_trees[i] = Or([and_node])
                    added_nodes.add(node_repr)
                elif not any(and_node == existing_node for existing_node in and_or_trees[i].children):
                            and_or_trees[i].children.append(and_node)

    return new_facts_added, and_or_trees

def is_complete_substitution(body, substitution):
    all_vars = set(var.name for term in body for var in term.arguments if isinstance(var, Variable))
    return all_vars.issubset(substitution.keys())


def substitute(term, substitution):
    if isinstance(term, Variable) and term.name in substitution:
        return substitution[term.name]
    elif isinstance(term, Term):
        substituted_arguments = tuple(substitute(arg, substitution) for arg in term.arguments)
        return Term(term.functor, substituted_arguments)
    else:
        return term

def construct_and_or_tree_node(clause, substitution):
    # Filter out 'add' predicate leaves
    children = [
        Leaf(substitute(term, substitution))
        for term in clause.body
        if term.functor != 'add'
    ]
    return And(children)


class Leaf:
    def __init__(self, term):
        self.term = term

    def __repr__(self):
        return f"Leaf({self.term})"

    def __eq__(self, other):
        return isinstance(other, Leaf) and self.term == other.term

class And:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        child_reprs = ', '.join([str(child) for child in self.children])
        return f"And({child_reprs})"

    def __eq__(self, other):
        if not isinstance(other, And) or len(self.children) != len(other.children):
            return False
        return all(c1 == c2 for c1, c2 in zip(sorted(self.children, key=repr), sorted(other.children, key=repr)))



class Or:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        child_reprs = ', '.join([str(child) for child in self.children])
        return f"Or({child_reprs})"



# Dummy example:

"""query = parse_term("addition(tensor(images,0), tensor(images,1), 0)")


Or = lambda x:  None
And = lambda x: None
Leaf = lambda x: None
and_or_tree = Or([
And([
    Leaf(parse_term("digit(tensor(images,0), 0)")),
    Leaf(parse_term("digit(tensor(images,1), 0)")),
])
])

return and_or_tree"""

