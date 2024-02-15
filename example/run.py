from nesy.model import NeSyModel, MNISTEncoder, CNN
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring, LukasieviczTNorm, GodelTNorm, ProductTNorm

import torch
import pytorch_lightning as pl

n_classes = 10

task_train = AdditionTask(n_classes=n_classes)
task_test = AdditionTask(n_classes=n_classes, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                logic_engine=ForwardChaining(),
                neural_predicates=neural_predicates,
                label_semantics=ProductTNorm(),
                n_digits = task_train.num_digits)

trainer = pl.Trainer(max_epochs=25)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=2),
            val_dataloaders=task_test.dataloader(batch_size=64))
