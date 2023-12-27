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
                 learning_rate = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics)

    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]]):
        # TODO: Note that you need to handle both the cases of single queries (List[Term]), like during training
        #  or of grouped queries (List[List[Term]]), like during testing.
        #  Check how the dataset provides such queries.


        # Check if the queries are single or grouped
        if isinstance(queries[0], Term):
            print("\n\n Single  query case \n\n")
            # Single queries case, typically during training
            and_or_trees = self.logic_engine.reason(self.program, queries)
            results = self.evaluator.evaluate(tensor_sources, and_or_trees, queries)
        else:
            print("\n\n Grouped query case \n\n")
            # Grouped queries case, typically during testing
            results = []
            for query_group in queries:
                print("\n\n Query group:\n\n", query_group)
                and_or_trees = self.logic_engine.reason(self.program, query_group)
                group_results = self.evaluator.evaluate(tensor_sources, and_or_trees, query_group)
                # Ensure group_results is 2D (1 row per group)
                group_results = group_results.unsqueeze(0)  # Adds a new dimension at position 0
                results.append(group_results)
            results = torch.cat(results, dim=0)  # Stacks along the new dimension

        print("\n\nresults: \n ", results, "\n")
        return results

    """Results should be of form:
        tensor([[1., 1., 1.],
        [1., 1., 1.]])

        and are right now of shape:
        Group results:
        tensor([[0.2327, 0.2629],
        [0.4998, 0.4997],
        [0.2675, 0.2374]]) """

    """

    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]]):
        # TODO: Note that you need to handle both the cases of single queries (List[Term]), like during training
        #  or of grouped queries (List[List[Term]]), like during testing.
        #  Check how the dataset provides such queries.
        and_or_tree = self.logic_engine.reason(self.program, queries)
        results = self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
        return results
        
    """

    def training_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1))
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
