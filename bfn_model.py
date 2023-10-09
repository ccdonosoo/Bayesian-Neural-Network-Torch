import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import typing
import tqdm

class BayesianFlowNetwork(nn.Module):
    def __init__(self, model, dim: int=2, n_classes: int=2, beta: float=1.0):
        """ Bayesian Flow Network

        Args:
            model (nn.Module: a neural network.
            dim (int, optional): dimension of data. Defaults to 2.
            n_classes (int, optional): n_classes. Defaults to 2.
            beta (float, optional): beta parameter. Defaults to 1.0.
        """
        super(BayesianFlowNetwork, self).__init__()
        self.beta = beta
        self.dim = dim
        self.n_classes = n_classes
        self.model = model      

    def forward(self, theta: Tensor, t: Tensor)->Tensor:
        """foward call of module

        Args:
            theta torch.Tensor: Tensor of shape (batch_size, dim, n_classes)
            this is a value in [0,1]
            t torch.Tensor: Tensor of shape (B,)

        Returns:
            torch.Tensor: out tensor (batch_size, dim, n_classes)
        """

        theta = (theta * 2) - 1  # scaled in [-1, 1]
        #0/0
        #theta = theta.view(theta.shape[0], -1)  # (B, D * K)
        #input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1)
        output = self.model(theta, t)  # (batch_size, dim, n_classes)
        output = output.view(output.shape[0], self.dim, -1)
        return output

    
    def discrete_output_distribution(self, theta, t):
        """Calculates the discrete output distribution.

        Args:
            theta torch.Tensor: Tensor of shape (batch_size, dim, n_classes)
            this is a value in [0,1]
            t torch.Tensor: Tensor of shape (B,)

        Returns:
            torch.Tensor: Output probability tensor. (batch_size, dim, n_classes)
        
        """
        batch_size, dim, n_classes = theta.shape
    
        # Get the forward pass output and reshape
        output = self.forward(theta, t)
    
        # Check the number of classes and compute the output probabilities accordingly 
        if n_classes == 2:
            p0_1 = torch.sigmoid(output)  # (B, D, 1)
            p0_2 = 1 - p0_1
            p0 = torch.cat((p0_1, p0_2), dim=-1)  # (B, D, 2)
        else:
            p0 = torch.nn.functional.softmax(output, dim=-1)
        return p0

    def process(self, x: Tensor):
        # Step 1: Sample t from U(0, 1)
        t = torch.rand((x.size(0),), device=x.device, dtype=torch.float32)

        # Step 2: Calculate Beta
        beta = self.beta * (t ** 2)  # (B,)

        # Step 3: Sample y from N(beta * (K * one_hot(X)) 
        one_hot_x = F.one_hot(x, num_classes=self.n_classes).float()  # (B, D, K)
        mean = beta[:, None, None] * (self.n_classes * one_hot_x - 1)
        #print("mean", mean.shape, mean)
        std = (beta * self.n_classes)[:, None, None].sqrt()
        #print("std", std.shape, std)
        eps = torch.randn_like(mean)
        y = mean + std * eps

        # Step 4: Compute the Theta
        theta = F.softmax(y, dim=-1)

        # Step 5: Calculate the output distribution
        p_0 = self.discrete_output_distribution(theta, t)  # (B, D, K)

        e_x = one_hot_x
        e_hat = p_0  # (B, D, K)
        L_infinity = self.n_classes * self.beta * t[:, None, None] * ((e_x - e_hat) ** 2)
        return L_infinity.mean()

    @torch.inference_mode()
    def sample(self, batch_size: int=128, nb_steps: int=10, device: str='cpu', eps_: float=1e-12):
        self.eval()  
        # get prior 
        theta = torch.ones((batch_size, self.dim, self.n_classes), device=device) / self.n_classes

        for i in range(1, nb_steps+1):
            t = (i-1) / nb_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)
            
            k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D)
            alpha = self.beta * (2 * i - 1) / (nb_steps ** 2)

            e_k = F.one_hot(k, num_classes=self.n_classes).float()  # (B, D, K)
            mean = alpha * (self.n_classes * e_k - 1)
            var = (alpha * self.n_classes)
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps  # (B, D, K)

            theta = F.softmax(y + torch.log(theta + eps_), dim=-1)


        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t))
        k_final = torch.distributions.Categorical(probs=k_probs_final).sample()

        return k_final
    
class BayesianFlowNetwork2D(nn.Module):
    def __init__(self, net, dim, n_classes, beta=3.0):
        """ Bayesian Flow Network for 2D data

        Args:
            model (nn.Module: a neural network.
            dim (int, optional): dimension of data. Defaults to 2.
            n_classes (int, optional): n_classes. Defaults to 2.
            beta (float, optional): beta parameter. Defaults to 1.0.
        """
        super(BayesianFlowNetwork2D, self).__init__()
        self.beta = beta
        self.dim = dim
        self.n_classes = n_classes

        self.net = net

    def forward(self,theta: Tensor, t: Tensor, ema)->Tensor:
        """foward call of module

        Args:
            theta torch.Tensor: Tensor of shape (batch_size, dim, n_classes)
            this is a value in [0,1]
            t torch.Tensor: Tensor of shape (B,)

        Returns:
            torch.Tensor: out tensor (batch_size, dim, n_classes)
        """
        theta = (theta * 2) - 1  # scaled in [-1, 1]
        theta = torch.transpose(theta, 1, 3)
        if ema is not None:
          with ema.average_parameters():
            output = self.net(theta + t[:, None, None, None])  # (B, D, D, K)
        else:
          output = self.net(theta + t[:, None, None, None])  # (B, D, D, K)

        return torch.transpose(output, 1, 3)

    def discrete_output_distribution(self,theta: Tensor, t: Tensor, ema=None)->Tensor:
        # Forward pass
        output = self.forward(theta, t, ema=ema)
        # Compute the output probabilities accordingly
        if self.n_classes == 2:
            p0_1 = torch.sigmoid(output)  # (B, D, D, 1)
            p0_2 = 1 - p0_1
            p0 = torch.cat((p0_1, p0_2), dim=-1)  # (B, D, D, 2)
        else:
            p0 = torch.nn.functional.softmax(output, dim=-1)
        return p0

    def process(self, x, t=None, training=True):
        # Step 1: Sample t from U(0, 1)
        if t is None:
          t = torch.rand((x.size(0),), device=x.device, dtype=torch.float32)
        else:
          t = torch.tensor(t, device=x.device, dtype=torch.float32)[None]
        # Step 2: Calculate Beta
        beta = self.beta * (t ** 2)  # (B,)
        # Step 3: Sample y from N(beta * (K * one_hot(X))
        one_hot_x = F.one_hot(x.permute(0, 2, 3, 1).to(torch.int64), num_classes=self.n_classes).float().squeeze()  # (B, D, D, K)
        mean = beta[:, None, None, None] * (self.n_classes * one_hot_x - 1)
        std = (beta * self.n_classes)[:, None, None, None].sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        # Step 4: Compute the Theta
        theta = F.softmax(y, dim=-1)
        # Step 5: Calculate the output distribution
        p_0 = self.discrete_output_distribution(theta, t)  # (B, D, D, K)

        e_x = one_hot_x
        e_hat = p_0  # (B, D, D, K)
        L_infinity = self.n_classes * self.beta * t[:, None, None, None] * ((e_x - e_hat) ** 2)
        if training:
          return L_infinity.mean()
        else:
           k = torch.distributions.Categorical(probs=p_0).sample()
           return L_infinity.mean(), y, k, t

    @torch.inference_mode()
    def sample(self, batch_size=128, nb_steps=10, ema=None, device='cpu'):
        self.eval()

        # get prior
        theta = torch.ones((batch_size, self.dim, self.dim, self.n_classes), device=device) / self.n_classes

        for i in tqdm(range(1, nb_steps+1)):
            t = (i-1) / nb_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)
            k_probs = self.discrete_output_distribution(theta, t, ema=ema)  # (B, D, D, K)
            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D, D)
            alpha = self.beta * (2 * i - 1) / (nb_steps ** 2)
            e_k = F.one_hot(k, num_classes=self.n_classes).float()  # (B, D, D, K)
            mean = alpha * (self.n_classes * e_k - 1)
            var = (alpha * self.n_classes)
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps  # (B, D, D, K)
            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)

        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t), ema=ema)
        k_final = torch.distributions.Categorical(probs=k_probs_final).sample()

        return k_final