from nesy.term import Term, Clause, Fact, Variable
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):
    """Class that represents the reasoning of our program, returns And-Or-Tree, proof of the provided queries

    Args:
        LogicEngine (class): Abstract class with reason method
    """

    def reason(self, program: tuple[Clause], queries: list[Term]):
        """Method to reason program and queries using forward chaining algorithm, 
           extracting new clauses form existing program and queries

        Args:
            program (tuple[Clause]): Logic program to specify the addition rules
            queries (list[Term]): queries asking for the probability which images sum up to a sum

        Returns:
            List[Term]: And-Or-Trees containing the proof for the queries
        """

        known_facts = set()

        for item in program:
            if isinstance(item, Fact):
                known_facts.add(item.term)

        # Initialize And-Or trees with placeholders
        and_or_trees = [None] * len(queries)

        # Apply forward chaining
        inferred = True
        while inferred:
            inferred = False
            for clause in program:
                if not isinstance(clause, Fact):
                    new_facts_added, and_or_trees = add_substitutions(clause, known_facts, queries, and_or_trees)
                    if new_facts_added:
                        inferred = True

        return and_or_trees

def add_substitutions(clause, known_facts, queries, and_or_trees):
    """Calculate possible substitutions and match them with queries

    Args:
        clause (String): addition claus
        known_facts (Set[Term]): All derived and known facts from the prolog program
        queries (list[Term]): queries asking for the probability which images sum up to a sum
        and_or_trees (List[Term]): Current Trees containing the proof for the queries

    Returns:
        Bool, List[Term]: boolean if there was inferred and extended And-Or-Trees
    """
    complete_substitutions = [{}]
    body = clause.body
    new_facts_added = False
    added_nodes = set()

    for term in body:
        new_substitutions = []
        for substitution in complete_substitutions:
            for fact in known_facts:
                if term.functor == fact.functor and len(term.arguments) == len(fact.arguments):
                    new_substitution = substitution.copy()
                    valid_substitution = True
                    for term_arg, fact_arg in zip(term.arguments, fact.arguments):
                        if isinstance(term_arg, Variable):
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

    for substitution in complete_substitutions:
        substituted_head = substitute(clause.head, substitution)
        if substituted_head not in known_facts:
            known_facts.add(substituted_head)
            new_facts_added = True
        for i, query in enumerate(queries):
            if substituted_head == query:
                and_node = construct_and_or_tree_node(clause, substitution)
                node_repr = repr(and_node)
                if and_or_trees[i] is None:
                    and_or_trees[i] = Or([and_node])
                    added_nodes.add(node_repr)
                elif not any(and_node == existing_node for existing_node in and_or_trees[i].children):
                            and_or_trees[i].children.append(and_node)

    return new_facts_added, and_or_trees

def is_complete_substitution(body, substitution):
    """Check for only complete substitutions

    Args:
        body (String): body of the clause
        substitution (dict): the substitution that is being checked

    Returns:
        bool: True if complete substitution
    """
    all_vars = set(var.name for term in body for var in term.arguments if isinstance(var, Variable))
    return all_vars.issubset(substitution.keys())


def substitute(term, substitution):
    """Perform the substitution

    Args:
        term (Term): Head of the clause
        substitution (dict): the substitution that is being checked

    Returns:
        Term: substituted term
    """
    if isinstance(term, Variable) and term.name in substitution:
        return substitution[term.name]
    elif isinstance(term, Term):
        substituted_arguments = tuple(substitute(arg, substitution) for arg in term.arguments)
        return Term(term.functor, substituted_arguments)
    else:
        return term

def construct_and_or_tree_node(clause, substitution):
    """Construct a node in the And-Or-Tree

    Args:
        clause (String): addition claus
        substitution (dict): the substitution that is being checked

    Returns:
        And: class that represents an Add leaf in the tree
    """
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