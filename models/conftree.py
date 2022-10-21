from .embedding.gin import GINEncoder
from .diffusion.score_model import ScoreNet, marginal_prob_std
import torch

class ConfTreeModel(torch.nn.Module):
	def __init__(self, args, num_convs=1):
		super(ConfTreeModel, self).__init__()
		
		self.embedding = GINEncoder(node_dim=args.in_node_features, edge_dim=args.in_edge_features,
									num_convs=1, activation='relu', short_cut=True)

		self.score_model = ScoreNet(args, hidden_dims=[128,64,32], 
									feature_dim=args.in_node_features)
	def forward(self, data, t):
		node_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		data.node_attr = self.embedding(node_attr, edge_index, edge_attr)
		assert data.node_attr.shape[0] == data.x.shape[0] # = Number of nodes
		score = self.score_model(data, t)
		return score