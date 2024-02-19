from nesy.term import Term, Clause, Fact, Variable
from abc import ABC, abstractmethod
import time
import logging

# Set up logging
#logging.basicConfig(filename='logicscale_11.log', level=logging.INFO)

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):
    """
    Class that represents the reasoning of our program, returns And-Or-Tree, proof of the provided queries

    Args:
        LogicEngine (class): Abstract class with reason method
    """

    def reason(self, program: tuple[Clause], queries: list[Term]):
        """
        Uses the forward chaining algorithm to reason over the given program and queries
        This method iteratively applies clauses to known facts to infer new facts until no more inferences can be made
        It constructs And-Or trees representing proofs for the queries

        Args:
            program (tuple[Clause]): Set of clauses defining the logical program
            queries (list[Term]): queries to be answered

        Returns:
            List[Term]: And-Or trees representing proofs for each query

        """
        #start_time = time.time()
        known_facts = set() # Stores the known facts that are derived from the program

        for item in program:
            if isinstance(item, Fact):
                known_facts.add(item.term)

        # Initialize And-Or trees for each query
        and_or_trees = [None] * len(queries)

        # Main loop for forward chaining, it continues until no new facts can be inferred
        inferred = True
        while inferred:
            inferred = False
            for clause in program:
                if not isinstance(clause, Fact):
                    # Attempt to add new facts based on the current clause and updated And-Or trees
                    new_facts_added, and_or_trees = add_substitutions(clause, known_facts, queries, and_or_trees)
                    if new_facts_added:
                        inferred = True

        #end_time = time.time()
        #elapsed_time = end_time - start_time

        #logging.info(f'The function took {elapsed_time:.4f} seconds to execute')

        return and_or_trees


# Calculates possible substitutions for a clause against known facts and updates And-Or trees
def add_substitutions(clause, known_facts, queries, and_or_trees):
    """
    Attempts to match the body of a clause with known facts to infer new facts (head of the clause) and
    updates the And-Or trees(proofs for the queries) accordingly

    Args:
        clause (Clause): Current clause being processed
        known_facts (Set[Term]): Set of known facts
        queries (List[Term]): List of queries being answered
        and_or_trees (List[Term]): Current state of the And-Or trees for each query

    Returns:
        Bool, List[Term]: Boolean indicating if new facts were added and the updated And-Or trees
    """
    new_facts_added = False # Indicates if new facts were inferred

    # Generate all possible substitutions(mappings) for the clause body based on known facts
    complete_substitutions = generate_substitutions(clause.body, known_facts)

    # Apply complete substitutions to infer new facts and update And-Or trees
    for substitution in complete_substitutions:

        # Apply the substitution to the clause's head
        substituted_head = substitute(clause.head, substitution)

        # If the inferred fact is not already known, add it to the known facts + mark that new facts were inferred
        if substituted_head not in known_facts:
            known_facts.add(substituted_head)
            new_facts_added = True

        # Check if the newly added fact is relevant to any of the queries
        for i, query in enumerate(queries):
            if substituted_head == query:
                # Update the And-Or tree for the query with the new fact
                and_node = construct_and_or_tree_node(clause, substitution)

                # Initialize the And-Or tree if it's the first fact for this query
                if and_or_trees[i] is None:
                    and_or_trees[i] = Or([and_node])

                # Otherwise, add new node to the existing tree, while avoiding duplicates
                elif not any(and_node == existing_node for existing_node in and_or_trees[i].children):
                    and_or_trees[i].children.append(and_node)

    return new_facts_added, and_or_trees

def generate_substitutions(body, known_facts):
    """
    Generates all possible substitutions that satisfy the body of a clause given the set of known facts

    Args:
        body (List[Term]): Body of the clause being processed
        known_facts (Set[Term]): The set of facts that are currently known

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary is a possible substitution that satisfies the clause's body with the known facts
    """
    # Start with a single empty substitution
    substitutions = [{}]

    # Iterate through each term in the clause's body
    for term in body:
        new_substitutions = []

        for substitution in substitutions:
            for fact in known_facts:
                # Check for potential match based on functor and argument count
                if match_terms(term, fact):

                    # Attempt to extend the substitution with the current fact
                    new_substitution = extend_substitution(substitution, term, fact)
                    if new_substitution is not None:
                        new_substitutions.append(new_substitution)

        # Update list of substitutions with the new ones
        substitutions = new_substitutions

    # Filter out incomplete substitutions
    return [sub for sub in substitutions if is_complete_substitution(body, sub)]



def match_terms(term, fact):
    """
    Checks if a term from a clause's body can match a known fact based on the functor and the number of arguments

    Args:
        term (Term): Term from the body of a clause
        fact (Term): A known fact

    Returns:
        bool: True if the term and the fact have the same functor and the same number of arguments, this indicates a potential match
    """
    return term.functor == fact.functor and len(term.arguments) == len(fact.arguments)



def extend_substitution(substitution, term, fact):
    """
    Attempts to extend a substitution by mapping variables in a term to corresponding terms in a fact

    Args:
        substitution (Dict): Current substitution being extended
        term (Term): Term from the body of a clause
        fact (Term): A known fact that matches the term

    Returns:
        Dict or None: An extended substitution if the term can be matched with the fact without conflicts, None otherwise
    """
    # Create a copy of the current substitution to attempt the extension
    new_substitution = substitution.copy()

    # Iterate through each argument pair from the term and the fact
    for term_arg, fact_arg in zip(term.arguments, fact.arguments):

        # If the term's argument is a variable, try to map it to the fact's argument
        if isinstance(term_arg, Variable):
            if term_arg.name in new_substitution and new_substitution[term_arg.name] != fact_arg:
                return None  # Conflicting substitution
            new_substitution[term_arg.name] = fact_arg

        # If the term's argument is not a variable, it must exactly match the fact's argument
        elif term_arg != fact_arg:
            return None

    # Return the extended substitution
    return new_substitution


def is_complete_substitution(body, substitution):
    """
    Determines if a substitution for the variables in the clause's body is complete

    Args:
        body (List[Term]): Body of the clause being processed
        substitution (dict): Maps variables to terms

    Returns:
        bool: True if the substitution is complete, otherwise False
    """
    # Extract all variables from the clause's body and check if they are all substituted
    all_vars = set(var.name for term in body for var in term.arguments if isinstance(var, Variable))
    return all_vars.issubset(substitution.keys())


def substitute(term, substitution):
    """
    Applies a substitution to a term, replacing variables with their corresponding terms

    Args:
        term (Term): Term/Variable to be substituted
        substitution (dict): The substitution mapping variables to terms

    Returns:
        Term: The term resulting from the substitution
    """
    # Substitute variables in the term or recursively substitute in arguments if the term is complex.
    if isinstance(term, Variable) and term.name in substitution:
        return substitution[term.name]
    elif isinstance(term, Term):
        substituted_arguments = tuple(substitute(arg, substitution) for arg in term.arguments)
        return Term(term.functor, substituted_arguments)
    else:
        return term

def construct_and_or_tree_node(clause, substitution):
    """
    Creates a node in the And-Or tree based on the given clause and substitution

    Args:
        clause (Clause): Clause being processed
        substitution (dict): The substitution applied to the clause

    Returns:
        And/Or: A node representing an AND operation in the And-Or tree
    """
    # Filter out 'add' predicate leaves, so to only have neural predicates as leaves
    children = [Leaf(substitute(term, substitution))for term in clause.body if term.functor != 'add']
    return And(children)


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