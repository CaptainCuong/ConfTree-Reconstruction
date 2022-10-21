import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        '''
        Args:
                x (torch.tensor): shape([N])
        Return:
                time_embedding (torch.tensor): shape([N,embed_dim])
        '''
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, args, hidden_dims=[128,64,32], feature_dim=44):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
                deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        self.t_embed_dim = hidden_dims[0]-feature_dim

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.t_embed_dim),
                 nn.Linear(self.t_embed_dim, self.t_embed_dim))
        
        # Encoding layers where the size decreases
        self.lin_lst = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.lin_lst.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
        self.lin_lst.append(nn.Linear(hidden_dims[-1], 4))

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = functools.partial(marginal_prob_std, sigma=args.sigma)
    
    def forward(self, data, t): 
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t)) # Shape (num_graph, t_embed_dim)
        batch_num_node = data.ptr[1:] - data.ptr[:-1]
        embed = torch.cat([embed[i].tile((num_node,1)) for i, num_node in enumerate(batch_num_node)], dim=0) # Shape (batch_num_node, t_embed_dim)
        node_attr = data.node_attr

        # Incorporate information from t
        node_attr = torch.cat([node_attr,embed], dim=-1) # Shape (batch_num_node, hidden_dims[0])
        h = node_attr
        for lin in self.lin_lst:
            h = lin(h)
            h = self.act(h)

        # Normalize output
        marginal_prob_std = self.marginal_prob_std(t)
        marginal_prob_std = torch.cat([marginal_prob_std[i].tile((num_node,1)) for i, num_node in enumerate(batch_num_node)], dim=0)
        h = h / marginal_prob_std
        return h

def marginal_prob_std(t, sigma):
    r"""Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    r"""Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    
    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)
