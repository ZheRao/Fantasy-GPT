from torch import nn
class loss_functions_class:
    def __init__(self):
        # dictionary for loss functions
        self.loss_dict = {
            "Binary Classification":nn.BCEWithLogitsLoss(),
            "Multiclass Classification":nn.CrossEntropyLoss(),
            "Linear": nn.HuberLoss(),
        }
        self.loss_description = {
            "Binary Classification": "Binary Cross Entropy Loss with a Sigmoid Layer",
            "Multiclass Classification": "Cross Entropy Loss with a Softmax Layer",
            "Linear": "Huber Loss - Combination of Squared Loss with Delta-Scaled L1 Loss"
        }
    # function to display all the loss functions here
    def __str__(self):
        for key in self.loss_description:
            print(f"\nInput: {key}, \nDescription: {self.loss_description[key]}\n")
        return f"\nThere are {len(self.loss_description)} Loss Functions Stored"

    # function to return loss function
    def get_loss_fn(self,problem: str):
        return self.loss_dict[problem]