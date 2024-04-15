import random
from nesy.parser import parse_program

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import product
from torch.utils.data import default_collate
import numpy as np
from itertools import product

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def add_noise_to_images(images, noise_level=0.1):
    """
    Adds noise to the images
    """
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0, 1)  # We ensure that pixel values are still in [0, 1]
    return noisy_images

def add_noise_to_labels(labels, n_classes=10, noise_rate=0.1):
    """
    Randomly changes a percentage of the labels to incorrect values
    """
    noisy_labels = labels.clone()
    n_noisy_labels = int(noise_rate * len(labels))
    noisy_indices = np.random.choice(len(labels), n_noisy_labels, replace=False)
    for idx in noisy_indices:
        # We assign a random class instead of the original label
        possible_labels = list(range(n_classes))
        possible_labels.remove(labels[idx].item())
        noisy_labels[idx] = torch.tensor(np.random.choice(possible_labels, 1))
    return noisy_labels


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])

def multi_digit_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2]), batch[3]


class AdditionTask(Dataset):
    """
    Class containing the MNIST addition task

    Args:
        Dataset (class): MNIST dataset
    """

    def __init__(self, n_addition=2, train=True, n_classes=10, nr_examples=None):
        self.train = train

        self.original_images = []
        self.original_targets = []

        # Collect all relevant images from the MNIST dataset
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)

        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)

        self.n_classes = n_classes
        self.n_addition = n_addition

        # Construct the logic program for the addition task
        self.program = self.construct_logic_program()

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.n_addition

    def construct_logic_program(self):
        """
        Constructs the logic program string that represents addition facts and queries.

        Returns:
            str: The constructed logic program as a string.
        """
        # Define the addition clause: of the form: addition(X,Y,Z) :- digit(X, N1), digit(Y, N2), add(N1, N2, Sum)
        addition_clause = "addition(" + ",".join(f"X{x}" for x in range(1, self.n_addition + 1)) + ",Sum) :- " + ", ".join(f"digit(X{x},N{x})" for x in range(1, self.n_addition + 1)) + ", add(" + ",".join(f"N{x}" for x in range(1, self.n_addition + 1)) + ",Sum).\n"

        # Define the facts of the form: add(N1, N2, Sum)
        addition_facts = "\n".join(f"add({', '.join(map(str, p))}, {sum(p)})." for p in product(range(self.n_classes), repeat=self.n_addition))

        # Construction of the neural predicates (weighted facts) of the form: nn(digit, tensor(images, 0), 0) :: digit(tensor(images, 0),0)
        neural_predicates = "\n".join(f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in product(range(self.n_addition), range(self.n_classes)))

        program_string = addition_clause + addition_facts + "\n" + neural_predicates
        #print(program_string)
        return parse_program(program_string)

    def __getitem__(self, index):
        images = self.original_images[index * self.n_addition: (index + 1) * self.n_addition]
        targets = self.original_targets[index * self.n_addition: (index + 1) * self.n_addition]
        target = int(targets.sum())

        # Construction of the queries
        # Queries are of the form: addition(tensor(images, 0), tensor(images, 1), Sum)

        if self.train:
            # Train phase
            # One query with the target sum for each group of images

            query_template = "addition({}, {})."
            tensor_indices = ', '.join("tensor(images, {})".format(i) for i in range(self.n_addition))
            query = parse_program(query_template.format(tensor_indices, target))[0].term

            tensor_sources = {"images": images}

            return tensor_sources, query, torch.tensor([1.0])
        else:
            # Validation phase
            # Group of queries (one query for each potential sum) for each group of images

            queries = [parse_program("addition({}, {}).".format(
            ', '.join("tensor(images, {})".format(i) for i in range(self.n_addition)), z))[0].term
            for z in range(self.n_classes * self.n_addition - (self.n_addition - 1))]

            tensor_sources = {"images": images}

            #print(queries)

            return tensor_sources, queries, target

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                        num_workers=num_workers)

    def __len__(self):
        return self.nr_examples


class NoisyAdditionTask(Dataset):
    """
    Dataset for MNIST addition task with optional noise addition to data and labels.
    """
    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None, apply_noise_to_data=False, apply_noise_to_labels=False, noise_level=0.2, noise_rate=0.1):
        self.train = train
        self.apply_noise_to_data = apply_noise_to_data
        self.apply_noise_to_labels = apply_noise_to_labels
        self.noise_level = noise_level
        self.noise_rate = noise_rate

        self.original_images = []
        self.original_targets = []

        for x, y in MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)

        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)

        self.n_classes = n_classes
        self.num_digits = n

        # Add noise to images
        if self.apply_noise_to_data:
            self.original_images = add_noise_to_images(self.original_images, self.noise_level)

        # Add noise to labels
        if self.apply_noise_to_labels:
            self.original_targets = add_noise_to_labels(self.original_targets, n_classes, self.noise_rate)

        self.program = self.construct_logic_program()

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def construct_logic_program(self):
        addition_clause = "addition(" + ",".join(f"X{x}" for x in range(1, self.num_digits + 1)) + ",Sum) :- " + ", ".join(f"digit(X{x},N{x})" for x in range(1, self.num_digits + 1)) + ", add(" + ",".join(f"N{x}" for x in range(1, self.num_digits + 1)) + ",Sum).\n"
        addition_facts = "\n".join(f"add({', '.join(map(str, p))}, {sum(p)})." for p in product(range(self.n_classes), repeat=self.num_digits))
        neural_predicates = "\n".join(f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in product(range(self.num_digits), range(self.n_classes)))
        program_string = addition_clause + addition_facts + "\n" + neural_predicates
        return parse_program(program_string)

    def __getitem__(self, index):
        images = self.original_images[index * self.num_digits: (index + 1) * self.num_digits]
        targets = self.original_targets[index * self.num_digits: (index + 1) * self.num_digits]
        target = int(targets.sum())

        # Construction of the queries
        # Queries are of the form: addition(tensor(images, 0), tensor(images, 1), Sum)

        if self.train:
            # Train phase
            # One query with the target sum for each group of images

            query_template = "addition({}, {})."
            tensor_indices = ', '.join("tensor(images, {})".format(i) for i in range(self.num_digits))
            query = parse_program(query_template.format(tensor_indices, target))[0].term

            tensor_sources = {"images": images}

            return tensor_sources, query, torch.tensor([1.0])
        else:
            # Validation phase
            # Group of queries (one query for each potential sum) for each group of images

            queries = [parse_program("addition({}, {}).".format(
            ', '.join("tensor(images, {})".format(i) for i in range(self.num_digits)), z))[0].term
            for z in range(self.n_classes * self.num_digits - (self.num_digits - 1))]

            tensor_sources = {"images": images}

            return tensor_sources, queries, target

    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate,
                        num_workers=num_workers)

    def __len__(self):
        return self.nr_examples


class MultiAdditionTask(Dataset):
    """
    Class containing the multi-digit MNIST addition task

    Args:
        Dataset (class): MNIST dataset
    """

    def __init__(self, n_addition=2, train=True, n_classes=10, n_multi=4, nr_examples=None):
        self.train = train

        self.original_images = []
        self.original_targets = []

        # Collect all relevant images from the MNIST dataset
        for x,y in  MNIST('data/MNIST/', train=train, download=True, transform=transform):
            if y < n_classes:
                self.original_images.append(x)
                self.original_targets.append(y)

        self.original_images = torch.stack(self.original_images)
        self.original_targets = torch.tensor(self.original_targets)

        self.n_classes = n_classes
        self.n_addition = n_addition
        self.n_multi = n_multi * 2

        # Construct the logic program for the addition task
        self.program = self.construct_logic_program()

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.n_multi

    def construct_logic_program(self):
        """
        Constructs the logic program string that represents addition facts and queries.

        Returns:
            str: The constructed logic program as a string.
        """
        half_n_multi = self.n_multi // 2  # Number of digits in each group

        # Dynamic creation of the addition clause using n_multi
        digits_x = [f"X{x+1}" for x in range(self.n_multi)]
        digits_n = [f"N{x+1}" for x in range(self.n_multi)]
        clause_parts = ", ".join(digits_x) + ", Sum"
        digit_conditions = ", ".join([f"digit({digits_x[i]}, {digits_n[i]})" for i in range(self.n_multi)])

        # Define how to pass digits directly to the add predicate
        first_half_args = ", ".join(digits_n[:half_n_multi])
        second_half_args = ", ".join(digits_n[half_n_multi:])
        add_statement = f"add({first_half_args}, {second_half_args}, Sum)"

        # Construct the addition clause
        addition_clause = f"addition({clause_parts}) :- {digit_conditions}, {add_statement}.\n"

        # Generate all possible addition facts
        addition_facts_set = set()
        for values in product(range(self.n_classes), repeat=self.n_multi):
            # Convert each half into a concatenated number
            first_half_value = int("".join(map(str, values[:half_n_multi])))
            second_half_value = int("".join(map(str, values[half_n_multi:])))
            sum_value = first_half_value + second_half_value
            values_args = ", ".join(map(str, values[:half_n_multi])) + ", " + ", ".join(map(str, values[half_n_multi:]))
            addition_facts_set.add(f"add({values_args}, {sum_value}).")

        # Convert the set back to a list for sorting or further manipulation
        addition_facts = list(sorted(addition_facts_set))

        # Construction of the neural predicates (weighted facts) of the form: nn(digit, tensor(images, 0), 0) :: digit(tensor(images, 0),0)
        neural_predicates = "\n".join(f"nn(digit, tensor(images, {x}), {y}) :: digit(tensor(images, {x}),{y})." for x, y in product(range(self.n_multi), range(self.n_classes)))

        program_string = addition_clause  + "\n".join(addition_facts) + "\n" + neural_predicates
        print(program_string)
        return parse_program(program_string)

    def generate_valid_sums(self):
        """
        Generates a list of all possible sums from combinations of up to n_digits.

        Args:
            n_classes (int): The number of digit classes (usually 10 for decimal digits).
            n_digits (int): The maximum number of digits to concatenate for sum calculations.

        Returns:
            List[int]: A list of all unique valid sums.
        """

        half_n_multi = self.n_multi // 2  # Number of digits in each group

        # Generate all combinations of numbers for the given number of digits

        # Calculate all possible numbers that can be formed by concatenating up to n_digits
        combinations = []
        for digits_count in range(1, half_n_multi + 1):
            for digits_tuple in product(range(self.n_classes), repeat=digits_count):
                number = int(''.join(map(str, digits_tuple)))
                combinations.append(number)

        # Calculate all possible sums from the combinations
        valid_sums = set()
        for num1 in combinations:
            for num2 in combinations:
                valid_sums.add(num1 + num2)

        #print("Valid sums: \n\n", valid_sums)

        return sorted(list(valid_sums))


    def __getitem__(self, index):
        images = self.original_images[index * self.n_multi: (index + 1) * self.n_multi]
        targets = self.original_targets[index * self.n_multi: (index + 1) * self.n_multi]

        # Calculate the number of digits to be combined
        half_n_multi = self.n_multi // 2

        # Convert groups of digits into integers
        combined_1 = sum([targets[i].item() * (10 ** (half_n_multi - i - 1)) for i in range(half_n_multi)])
        combined_2 = sum([targets[i + half_n_multi].item() * (10 ** (half_n_multi - (i % half_n_multi) - 1)) for i in range(half_n_multi)])
        target = int(combined_1 + combined_2)
        #print("Target: \n\n", target)

        # Construction of the queries
        # Queries are of the form: addition(tensor(images, 0), tensor(images, 1), Sum)

        tensor_sources = {"images": images}

        if self.train:
            # Train phase
            # One query with the target sum for each group of images

            query_template = "addition({}, {})."
            tensor_indices = ', '.join("tensor(images, {})".format(i) for i in range(self.n_multi))
            query = parse_program(query_template.format(tensor_indices, target))[0].term

            return tensor_sources, query, torch.tensor([1.0]), 0
        else:
            # Validation phase
            # Group of queries (one query for each potential sum) for each group of images

            # Dynamically generate valid sums based on the combination logic
            valid_sums = self.generate_valid_sums()

            queries = [parse_program("addition({}, {}).".format(
            ', '.join("tensor(images, {})".format(i) for i in range(self.n_multi)), z))[0].term
            for z in valid_sums]

            #print("Queries: \n\n", queries)

            return tensor_sources, queries, target, valid_sums


    def dataloader(self, batch_size=2, shuffle=None, num_workers=0):
        if shuffle is None:
            shuffle = self.train

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=multi_digit_collate,
                        num_workers=num_workers)

    def __len__(self):
        return self.nr_examples
