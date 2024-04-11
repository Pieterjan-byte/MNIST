from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask, NoisyAdditionTask, MultiAdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring, LukasieviczTNorm, GodelTNorm, ProductTNorm
import time
import logging

# Set up logging
#logging.basicConfig(filename='digitsscale_5.log', level=logging.INFO)

import torch
import pytorch_lightning as pl

# Define the number of classes of possible digits(n_classes = 2 means only add images representing 0s and 1s), 1 < n_classes < 11
n_classes = 2

# Define the number of single digits number we are summing, 1 < n_addition < 10
n_addition = 2

n_multi = 2

#start_time = time.time()

task_train = AdditionTask(n=n_addition, n_classes=n_classes)
task_test = AdditionTask(n=n_addition, n_classes=n_classes, train=False)

# To add noise to input data or labels use this addition task
#task_train = NoisyAdditionTask(n=n_addition, n_classes=n_classes, apply_noise_to_data=True, apply_noise_to_labels=True, noise_level=0.1)



# Define your parameters such as max_digits
max_digits = 2  # Example for 2-digit addition

#task_train = MultiAdditionTask(n=n_addition, n_classes=n_classes, train=True)
#task_test = MultiAdditionTask(n=n_addition, n_classes=n_classes, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                logic_engine=ForwardChaining(),
                neural_predicates=neural_predicates,
                label_semantics=SumProductSemiring(),
                n_digits = task_train.num_digits)

# Define the number of epochs we use to train the neural network
n_epochs = 5

# Define the batch size for training
train_batch_size = 2

# Define the batch size for validation
val_batch_size = 8

trainer = pl.Trainer(max_epochs=n_epochs)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=train_batch_size),
            val_dataloaders=task_test.dataloader(batch_size=val_batch_size))

#end_time = time.time()
#elapsed_time = end_time - start_time

#logging.info(f'The system took {elapsed_time:.4f} seconds to execute')
