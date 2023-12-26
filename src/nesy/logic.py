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

        # Function to match and instantiate a clause with a query
        def instantiate_clause(clause, query):
            substitution = {}
            for (clause_arg, query_arg) in zip(clause.head.arguments, query.arguments):
                substitution[str(clause_arg)] = str(query_arg)
            instantiated_body = []
            for term in clause.body:
                instantiated_term = Term(term.functor, [substitution.get(str(arg), str(arg)) for arg in term.arguments])
                instantiated_body.append(instantiated_term)
            return instantiated_body

        print("\n\nProgram: \n", program)
        print("\n\nQueries: \n", queries)

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

        print("\n\nKnown facts: \n", known_facts)

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

        print("\n\nKnown facts after forward chaining: \n", known_facts)

        # Build the And-Or tree for each query
        and_or_trees = []
        for query in queries:
            relevant_clauses = [clause for clause in program if isinstance(clause, Clause) and clause.head.functor == query.functor and len(clause.head.arguments) == len(query.arguments)]
            and_nodes = []
            for clause in relevant_clauses:
                instantiated_body = instantiate_clause(clause, query)
                and_nodes.append(And([Leaf(term) for term in instantiated_body]))

            or_node = Or(and_nodes)
            and_or_trees.append(or_node)

        print("\n\n And or trees: \n", and_or_trees)

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

