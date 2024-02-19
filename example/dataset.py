from nesy.parser import parse_program

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from itertools import permutations, product
from torch.utils.data import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def custom_collate(batch):
    batch = tuple(zip(*batch))
    return default_collate(batch[0]), batch[1], default_collate(batch[2])


class AdditionTask(Dataset):
    """
    Class containing the MNIST addition task

    Args:
        Dataset (class): MNIST dataset
    """

    def __init__(self, n=2, train=True, n_classes=10, nr_examples=None):
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
        self.num_digits = n

        # Construct the logic program for the addition task
        self.program = self.construct_logic_program()

        if nr_examples is not None:
            if nr_examples > self.nr_examples:
                raise ValueError('nr_examples exceeds to number of available examples in this dataset')
            else:
                self.nr_examples = nr_examples
        else:
            self.nr_examples = len(self.original_images) // self.num_digits

    def construct_logic_program(self):
        """
        Constructs the logic program string that represents addition facts and queries.

        Returns:
            str: The constructed logic program as a string.
        """
        # Define the addition clause: of the form: addition(X,Y,Z) :- digit(X, N1), digit(Y, N2), add(N1, N2, Sum)
        addition_clause = "addition(" + ",".join(f"X{x}" for x in range(1, self.num_digits + 1)) + ",Sum) :- " + ", ".join(f"digit(X{x},N{x})" for x in range(1, self.num_digits + 1)) + ", add(" + ",".join(f"N{x}" for x in range(1, self.num_digits + 1)) + ",Sum).\n"

        # Define the facts of the form: add(N1, N2, Sum)
        addition_facts = "\n".join(f"add({', '.join(map(str, p))}, {sum(p)})." for p in product(range(self.n_classes), repeat=self.num_digits))

        # Construction of the neural predicates (weighted facts) of the form: nn(digit, tensor(images, 0), 0) :: digit(tensor(images, 0),0)
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
