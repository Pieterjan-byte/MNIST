from typing import List, Dict
import torch
import pytorch_lightning as pl

from nesy.semantics import Semantics
from nesy.term import Clause, Term
from nesy.logic import LogicEngine
from torch import nn
from sklearn.metrics import accuracy_score
from nesy.evaluator import Evaluator

import logging

# Set up logging
logging.basicConfig(filename='output.log', level=logging.INFO)

class MNISTEncoder(nn.Module):
    def __init__(self, n):
        self.n = n
        super(MNISTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 30),
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
        """
        Forward step of the Neural Network

        Args:
            tensor_sources (Dict[str, torch.Tensor]): Dictionary containing the MNIST Images in the form of tensor data
            queries (list[Term]): queries asking for the probability which images sum up to a sum

        Returns:
            torch.Tensor: Output tensor containing probabilities
        """
        # Set up logging
        logging.basicConfig(filename='output.log', level=logging.INFO)

        # Check if the queries are single or grouped
        if isinstance(queries[0], Term):
            single_query = True
            and_or_trees = self.logic_engine.reason(self.program, queries, single_query)
            results = self.evaluator.evaluate(tensor_sources, and_or_trees, i=0)
            logging.info(queries)
            logging.info("\n")
            logging.info(and_or_trees)
            logging.info("\n")
            logging.info(results)
            logging.info("\n")
        else:
            single_query = False
            results = []
            # i contains the index of the query group in queries, to know which images to work with
            i = 0
            for query_group in queries:
                and_or_trees = self.logic_engine.reason(self.program, query_group, single_query)
                group_results = self.evaluator.evaluate(tensor_sources, and_or_trees, i)
                group_results = group_results.unsqueeze(0) # Adjust shape for concatenation
                results.append(group_results)
                i += 1
            results = torch.cat(results, dim=0)

        return results

    def training_step(self, I, batch_idx):
        """
        Performs a single training step

        Args:
            I (tuple): Contains tensor sources, queries, and true labels for the batch.

        Returns:
            float: The computed loss for the training step.
        """
        # Set up logging
        logging.basicConfig(filename='output.log', level=logging.INFO)
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        logging.info(y_true)
        logging.info("\n")
        logging.info(y_preds)
        logging.info("\n")
        logging.info("---------------------------------------")

        return loss


    def validation_step(self, I, batch_idx):
        """
        Performs a single validation step

        Args:
            I (tuple): Contains tensor sources, queries, and true labels for the batch.

        Returns:
            float: The computed accuracy for the validation step.
        """
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        #print("Y preds: \n\n ", y_preds)
        accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1))
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer