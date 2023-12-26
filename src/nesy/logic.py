from nesy.term import Term, Clause, Fact
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):

    def reason(self, program: tuple[Clause], queries: list[Term]):
        # TODO: Implement this

        # Initialize known facts
        known_facts = set()

        # Update to handle Fact objects
        for item in program:
            if isinstance(item, Fact):
                # Treat Fact as a Clause with an empty body
                known_facts.add(item.term)
            elif isinstance(item, Clause) and not item.body:
                # Handle empty-body Clauses as facts
                known_facts.add(item.head)

        # Apply forward chaining
        inferred = True
        while inferred:
            inferred = False
            for item in program:
                if isinstance(item, Clause):
                    if all(term in known_facts for term in item.body):
                        if item.head not in known_facts:
                            known_facts.add(item.head)
                            inferred = True

        # Build the And-Or tree for each query
        and_or_trees = []
        for query in queries:
            # Find clauses relevant to the query
            relevant_clauses = [clause for clause in program if isinstance(clause, Clause) and query.functor in clause.head]

            # Create 'And' nodes for each relevant clause
            and_nodes = [And([Leaf(term) for term in clause.body]) for clause in relevant_clauses]

            # Create an 'Or' node with all 'And' children
            or_node = Or(and_nodes)

            and_or_trees.append(or_node)

        print(and_or_trees)

        return and_or_trees

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

class Leaf:
    def __init__(self, term):
        self.term = term

    def __repr__(self):
        return f"Leaf({self.term})"

class And:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        child_reprs = ', '.join([str(child) for child in self.children])
        return f"And({child_reprs})"

class Or:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        child_reprs = ', '.join([str(child) for child in self.children])
        return f"Or({child_reprs})"

