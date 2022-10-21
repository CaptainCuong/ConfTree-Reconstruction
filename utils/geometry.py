import torch
from torch_scatter import scatter_add
import copy

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1).reshape(-1,1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_cos_angle(pos, vec1, vec2):
    """
    Args:
        pos:  (N, 3)
        vec1:  (2, A), u1 --> v1, 
        vec2:  (2, A), u2 --> v2,
    """
    u1, v1 = vec1
    u2, v2 = vec2
    vec1 = pos[v1] - pos[u1] # (A, 3)
    vec2 = pos[v2] - pos[u2]
    inner_prod = torch.sum(vec1 * vec2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(vec1, dim=-1, keepdim=True) * torch.norm(vec2, dim=-1, keepdim=True)   # (A, 1)
    cos_angle = inner_prod / length_prod    # (A, 1)
    
    assert torch.isnan(cos_angle).sum().item() == 0
    return cos_angle

def get_pseudo_vec(pos, vec1, vec2):
    """
    Args:
        pos:  (N, 3)
        vec1:  (2, E), u1 --> v1, 
        vec2:  (2, E), u2 --> v2, 
    """
    u1, v1 = vec1
    u2, v2 = vec2
    vec1 = pos[v1] - pos[u1] # (E, 3)
    vec2 = pos[v2] - pos[u2] # (E, 3)

    return torch.cross(vec1, vec2, dim=1)

def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]   # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)    # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)    # (A, 1)
    
    assert torch.isnan(dihedral).sum().item() == 0
    return dihedral

def atfc_bond_sphe_pos(data):
    '''
    Add artificial bonds and spherical positions
    Args:
        data: (torch_geometric.data.data.Data)
            attrs: ['x', 'edge_index', 'edge_attr', 'z', 'canonical_smi', 'mol', 'pos', 'weights', 'tree']
    '''
    num_node = data.tree.number_of_nodes()
    p1, N = torch.tensor([[u,v] for u,v in data.tree.edges()]).T
    p1, N = p1.tolist(), N.tolist()

    p2 = []
    for n in p1:
        if len(list(data.tree.predecessors(n))) == 1:
            p2.append(list(data.tree.predecessors(n))[0])
        elif len(list(data.tree.predecessors(n))) == 0:
            p2.append(-1)
        else:
            raise Exception('Not Tree')
        
    p3 = []
    for n in p2:
        if n == -1:
            p3.append(-2)
        elif len(list(data.tree.predecessors(n))) == 1:
            p3.append(list(data.tree.predecessors(n))[0])
        elif len(list(data.tree.predecessors(n))) == 0:
            p3.append(-1)
        else:
            raise Exception('Not Tree')

    # Artificial Bond
    '''
    edge_attr
    tensor([[0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.]])
    edge_index
    tensor([[0, 1, 0, 3, 1, 2],
            [1, 0, 3, 0, 2, 1]])
    '''
    add_edge = [data.edge_index]
    add_edge_feature = [data.edge_attr]

    for u,v in list(zip(p2,N)):
        if u < 0: continue
        exist = False
        for i, e in enumerate(data.edge_index.T):
            if [u,v] == e.tolist():
                data.edge_attr[i][4] = 1
                exist = True
        if not exist:
            add_edge.append(torch.tensor([[u],[v]]))
            add_edge_feature.append(torch.tensor([[0, 0, 0, 0, 1, 0]]))
    
    for u,v in list(zip(p3,N)):
        if u < 0: continue
        exist = False
        for i, e in enumerate(data.edge_index.T):
            if [u,v] == e.tolist():
                data.edge_attr[i][5] = 1
                exist = True
        if not exist:
            add_edge.append(torch.tensor([[u],[v]]))
            add_edge_feature.append(torch.tensor([[0, 0, 0, 0, 0, 1]]))
    
    data.edge_index = torch.cat(add_edge, dim=1)
    data.edge_attr = torch.cat(add_edge_feature, dim=0)

    # Spherical Postion
    sphe_pos = []
    pos_3D = []
    for pos in data.pos:
        
        # Artificial starting points
        pos = pos - pos[0]
        psuedo_node = torch.tensor([[0.82,0.82,0.82],[0,1.23,0]])
        pos = torch.cat([psuedo_node, pos], dim=0)
        n1, n2, n3, n4 = [i+2 for i in N], [i+2 for i in p1], [i+2 for i in p2], [i+2 for i in p3] 
        n5 = [i for i in range(num_node+2, num_node*2+1)] # Add (num_node-1) third vertex
        # n2 - x, n3 - O, n4 - y, n5 - z
        
        
        # z axis and z coordinate
        third_vec = get_pseudo_vec(pos, [n3,n2], [n3,n4])
        third_point = third_vec + pos[n3]
        pos = torch.cat([pos, third_point], axis = 0)

        # calculate cos(bond_angle)
        cos_bond_angle_x = get_cos_angle(pos, [n2,n1], [n3,n2])
        cos_bond_angle_y = get_cos_angle(pos, [n2,n1], [n3,n4])
        try:
            cos_bond_angle_z = get_cos_angle(pos, [n2,n1], [n3,n5])
        except:
            continue
        length = get_distance(pos, [n2,n1])  
        
        feature = torch.cat((cos_bond_angle_x, cos_bond_angle_y, cos_bond_angle_z, length), dim = -1)
        feature = torch.cat([torch.tensor([[0.5,0.5,0.5,1.23]]), feature], dim = 0)
        assert feature.shape[0] == data.tree.number_of_nodes(), str(feature.shape[0]) + '  ' + str(data.tree.number_of_nodes()) + '  ' + str(len(n2))
        sphe_pos.append(feature)
        pos_3D.append(pos[2:,:])

    data.sphe_pos = sphe_pos # Shape (N, 4)
    data.pos = pos_3D   # Shape (2N+2, 3)
    assert len(data.sphe_pos) == len(data.pos)
    
