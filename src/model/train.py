import torch

class Train:
    @staticmethod
    def train_loop(num_of_epoch: int, input_data: torch.autograd.Variable, ground_truth: torch.autograd.Variable,
                   optimizer: torch.optim.Adam, model: torch.nn.Sequential):
        """A simple train loop.

                Args:
                    num_of_epoch (int): Number of epoch.
                    input_data (torch.autograd.Variable): Input data.
                    ground_truth(torch.autograd.Variable):  Ground truth.
                    optimizer (torch.optim.Adam): ADAM optimizer
                    model(torch.nn.Sequential): Neural network model.

        """
        loss_fn = torch.nn.MSELoss(reduction='sum')

        for t in range(num_of_epoch):
            output_pred = model(input_data)
            loss = loss_fn(output_pred, ground_truth)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
