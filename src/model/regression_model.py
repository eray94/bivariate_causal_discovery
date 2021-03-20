import torch
from src.constant import ModelConstants


class RegressionModel:
    @staticmethod
    def build_model(input_dim=1, hidden_dim=20, output_dim=1) -> torch.nn.Sequential:
        """Build simple NN model with one hidden layer for regression problem.

               Args:
                   input_dim (int): Input dimension of NN.
                   hidden_dim (int): Number of neurons in hidden layer.
                   output_dim (int): Output dimension of NN
               Returns:
                   torch.nn.Sequential: Neural network model.
       """
        model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                    torch.nn.LayerNorm(hidden_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_dim, output_dim))

        return model

    @staticmethod
    def init_weights(model: torch.nn.Sequential):
        """Initialize weights with xavier initialization.

               Args:
                   model (int): Input dimension of NN.
         """

        if type(model) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)

    @staticmethod
    def get_num_of_epoch(train_size: int) -> int:
        """Decide number of epochs. A bit higher for larger a dataset.

               Args:
                   train_size (int): Size of train data.
               Returns:
                   int: Neural Number of epoch.
         """
        if train_size <= ModelConstants.EPOCH_THRESHOLD:
            num_of_epoch = ModelConstants.NUM_OF_EPOCH_SMALL
        else:
            num_of_epoch = ModelConstants.NUM_OF_EPOCH_LARGE

        return num_of_epoch

    @staticmethod
    def get_residuals(model: torch.nn.Sequential, test_input: torch.autograd.Variable,
                      ground_truth: torch.autograd.Variable):
        """Calculate residuals of regression.

                Args:
                    model(torch.nn.Sequential): Neural network model.
                    test_input (torch.autograd.Variable): Input test data.
                    ground_truth(torch.autograd.Variable):  Ground truth of test data.
                Returns:
                   torch.autograd.Variable: Residuals.
        """
        with torch.no_grad():
            test_pred = model(test_input)
            res = test_pred - ground_truth
        return res
