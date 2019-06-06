class Trainer:

    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers

    def prepare_training(self):
        """ puts models in training mode and resets optimizers """

        for model in self.models:
            model.train()
        for opt in self.optimizers:
            opt.zero_grad()

    def do_backward(self, loss):
        """ does the backward pass """

        loss.backward()
        for opt in self.optimizers:
            opt.step()

    def prepare_evaluation(self):
        """ sets models in evaluation mode """

        for opt in self.optimizers:
            opt.zero_grad()
        for model in self.models:
            model.eval()
