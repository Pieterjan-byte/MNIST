from typing import List, Dict
import torch
import pytorch_lightning as pl

import nesy.parser
from nesy.semantics import Semantics
from nesy.term import Clause, Term
from nesy.logic import LogicEngine
from torch import nn
from sklearn.metrics import accuracy_score
from nesy.evaluator import Evaluator
import math

class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        init convolution and activation layers
        Args:
            input_size: (1,28,28)
            num_classes: 10
        """
        self.n = num_classes
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, num_classes)


    def forward(self, x):
        """
        forward function describes how input tensor is transformed to output tensor
        Args:
            x: (Nx1x28x28) tensor
        """
        #We flatten the tensor
        original_shape = x.shape
        n_dims = len(original_shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        o = self.fc1(x)
        #We restore the original shape
        o = o.view(*original_shape[0:n_dims-3], self.n)
        return o


class MNISTEncoder(nn.Module):
    def __init__(self, n):
        self.n = n
        super(MNISTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, n),
            nn.Softmax(-1))


    def forward(self, x):
        #We flatten the tensor
        original_shape = x.shape
        n_dims = len(original_shape)
        x = x.view(-1, 784)
        o =  self.net(x)

        #We restore the original shape
        o = o.view(*original_shape[0:n_dims-3], self.n)
        return o

class NeSyModel(pl.LightningModule):


    def __init__(self, program : List[Clause],
                neural_predicates: torch.nn.ModuleDict,
                logic_engine: LogicEngine,
                label_semantics: Semantics,
                n_digits: int,
                learning_rate = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics)
        self.n_digits = n_digits

    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]]):
        # TODO: Note that you need to handle both the cases of single queries (List[Term]), like during training
        #  or of grouped queries (List[List[Term]]), like during testing.
        #  Check how the dataset provides such queries.
        """Forward step of the Neural Network

        Args:
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensors
            queries (list[Term]): queries asking for the probability which images sum up to a sum

        Returns:
            torch.Tensor: Tensor containing probabilities
        """

        # Check if the queries are single or grouped
        if isinstance(queries[0], Term):
            and_or_trees = self.logic_engine.reason(self.program, queries)
            results = self.evaluator.evaluate(tensor_sources, and_or_trees, queries, i=0)
        else:
            results = []
            i = 0
            for query_group in queries:
                and_or_trees = self.logic_engine.reason(self.program, query_group)
                group_results = self.evaluator.evaluate(tensor_sources, and_or_trees, query_group, i)
                group_results = group_results.unsqueeze(0)
                results.append(group_results)
                i += 1
            results = torch.cat(results, dim=0)

        return results

    def training_step(self, I, batch_idx):
        """Loss calculation during training step

        Args:
            I (tensor_sources, queries, torch.tensor): Indexation on the AdditionTask class (getitem)
            batch_idx (int): batch counter

        Returns:
            float: The loss calculated for this Training step
        """
        tensor_sources, queries, y_true = I
        print(batch_idx)
        y_preds = self.forward(tensor_sources, queries)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, I, batch_idx):
        """Accuracy calculation during validation step

        Args:
            I (tensor_sources, queries, torch.tensor): Indexation on the AdditionTask class (getitem)
            batch_idx (int): batch counter

        Returns:
            float: The accuracy calculated during the Validation step
        """
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1))
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer