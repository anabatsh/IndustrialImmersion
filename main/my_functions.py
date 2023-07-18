import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from fairseq_dropout import FairseqDropout


atom_types_number = 10
bond_types_number = atom_types_number ** 2 + 1

def integer_symmetric_matrix(v, h, w):
    m = torch.randint(v, size=(h, w))
    m = torch.tril(m, 0) + torch.tril(m, -1).T
    return m

bond_types = integer_symmetric_matrix(bond_types_number, atom_types_number, atom_types_number)

def generate_atoms_types(atoms_number):
    return torch.randint(atom_types_number, size=(atoms_number,))

def generate_atoms_features(atoms_number, atom_feature_dim):
    return torch.randn(atoms_number, atom_feature_dim)

def generate_atoms_positions(atoms_number, std=1.0, mu_std=0.0):
    pos = [torch.rand(3)]
    for _ in range(atoms_number - 1):
        pos.append((torch.rand(3) - 0.5) * 2 * std + pos[-1])
    pos = torch.vstack(pos)
    pos -= pos.mean(0)
    pos += torch.randn(3) * mu_std
    return pos

def generate_bonds(pos_from, pos_to, atoms_types_from, atoms_types_to, max_degree, threshold):
    pairwise_dists = torch.cdist(pos_from, pos_to)
    pairwise_dists_tril = torch.tril(pairwise_dists)
    pairwise_dists_tril[pairwise_dists_tril == 0.0] = threshold
    pairwise_dists_sorted, pairwise_dists_sorted_idx = torch.sort(pairwise_dists_tril)
    pairwise_dists_sorted = pairwise_dists_sorted[:, :max_degree]
    pairwise_dists_sorted_idx = pairwise_dists_sorted_idx[:, :max_degree]
    mask = pairwise_dists_sorted < threshold
    atoms_from = torch.where(mask)[0]
    atoms_to = pairwise_dists_sorted_idx[mask]
    atoms_from_to = torch.stack([atoms_from, atoms_to])
    dists = pairwise_dists[atoms_from, atoms_to]
    bonds = bond_types[atoms_types_from[atoms_from], atoms_types_to[atoms_to]]
    return atoms_from_to, dists, bonds

def generate_complex(
    protein_atoms_number,
    ligand_atoms_number,
    atom_feature_dim,
    max_degree, threshold,
    std=1.0, mu_std=0.0
):
    data = HeteroData()
    
    data['protein'].atoms = generate_atoms_types(protein_atoms_number)
    data['protein'].x = generate_atoms_features(protein_atoms_number, atom_feature_dim)
    data['protein'].pos = generate_atoms_positions(protein_atoms_number, std)

    data['ligand'].atoms = generate_atoms_types(ligand_atoms_number)
    data['ligand'].x = generate_atoms_features(ligand_atoms_number, atom_feature_dim)
    data['ligand'].pos = generate_atoms_positions(ligand_atoms_number, std, mu_std=mu_std)

    # protein-protein
    protein_protein, dists, bonds = generate_bonds(
        data['protein'].pos, data['protein'].pos,
        data['protein'].atoms, data['protein'].atoms,
        max_degree, threshold
    )
    data['protein', '-', 'protein'].edge_index = protein_protein
    data['protein', '-', 'protein'].edge_attr = (dists, bonds)

    # ligand-ligand
    ligand_ligand, dists, bonds = generate_bonds(
        data['ligand'].pos, data['ligand'].pos, 
        data['ligand'].atoms, data['ligand'].atoms, 
        max_degree, threshold
    )
    data['ligand', '-', 'ligand'].edge_index = ligand_ligand
    data['ligand', '-', 'ligand'].edge_attr = (dists, bonds)
    
    # protein-ligand
    protein_ligand, dists, bonds = generate_bonds(
        data['protein'].pos, data['ligand'].pos, 
        data['protein'].atoms, data['ligand'].atoms, 
        max_degree, threshold
    )
    data['protein', '-', 'ligand'].edge_index = protein_ligand
    data['protein', '-', 'ligand'].edge_attr = (dists, bonds)

    data['rmsd'] = dists.mean().item() if len(dists) else 0.0
    return data

def show_complex_networx(
        data,
        field_name='x',
        ligand_name='ligand',
        protein_name='protein',
        ligand_bond_name='-',
        protein_bond_name='-',
        protein_ligand_bond_name='-',
        with_labels=True,
        ax=None
    ):
    protein_nodes = range(data[protein_name][field_name].shape[0])
    ligand_nodes = range(data[ligand_name][field_name].shape[0])

    protein_protein_edges = data[protein_name, protein_bond_name, protein_name].edge_index.numpy().T
    ligand_ligand_edges = data[ligand_name, ligand_bond_name, ligand_name].edge_index.numpy().T
    protein_ligand_edges = data[protein_name, protein_ligand_bond_name, ligand_name].edge_index.numpy().T

    G = nx.Graph()

    for node in protein_nodes:
        G.add_node(f'{node}'+r'$_p$', label=node, color='pink')
    for edge in protein_protein_edges:
        edge = [f'{node}'+r'$_p$' for node in edge]
        G.add_edge(*edge, color='pink')

    for node in ligand_nodes:
        G.add_node(f'{node}'+r'$_l$', label=node, color='lightskyblue')
    for edge in ligand_ligand_edges:
        edge = [f'{node}'+r'$_l$' for node in edge]
        G.add_edge(*edge, color='lightskyblue')

    for edge in protein_ligand_edges:
        edge = [f'{edge[0]}'+r'$_p$', f'{edge[1]}'+r'$_l$']
        G.add_edge(*edge, color='black')

    node_colors = nx.get_node_attributes(G,'color').values()
    edge_colors = nx.get_edge_attributes(G,'color').values()
    node_labels = nx.get_node_attributes(G,'label')
    node_sizes = [v * 20 for _, v in nx.degree(G)]

    if ax is None:
        plt.figure(figsize=(4, 4))
    nx.draw(G, ax=ax, node_color=node_colors, edge_color=edge_colors, 
            node_size=node_sizes, font_size=8, labels=node_labels, with_labels=with_labels)
    if ax is None:
        plt.show()
        
def show_complex_batch_networx(
        batch, 
        field_name='x',
        ligand_name='ligand', 
        protein_name='protein', 
        ligand_bond_name='-', 
        protein_bond_name='-',
        protein_ligand_bond_name='-',
        with_labels=True, 
    ):
    n_5 = int(np.ceil(len(batch) / 5))
    _, axes = plt.subplots(n_5, 5, figsize=(15, 3*n_5))
    for data, ax in zip(batch, axes):
        show_complex_networx(data, field_name, ligand_name, protein_name, 
                             ligand_bond_name, protein_bond_name, protein_ligand_bond_name, 
                             with_labels, ax=ax)
        ax.axis('on')
    plt.show()

class ProteinLigandComplexes(Dataset):
    def __init__(
        self,
        mode='train',
        N=1000,
        protein_atoms_number_max=20, 
        ligand_atoms_number_max=10,
        atom_feature_dim=8,
        max_degree=5, threshold=1.8,
        std=1.0, mu_std=2.0
    ):
        self.N = N
        self.protein_atoms_number_max = protein_atoms_number_max
        self.ligand_atoms_number_max = ligand_atoms_number_max
        self.atom_feature_dim = atom_feature_dim
        self.max_degree = max_degree
        self.threshold = threshold
        self.std = std
        self.mu_std = mu_std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        protein_atoms_number = torch.randint(1, self.protein_atoms_number_max, (1,))
        ligand_atoms_number = torch.randint(1, self.ligand_atoms_number_max, (1,))
        data = generate_complex(
            protein_atoms_number, ligand_atoms_number, 
            self.atom_feature_dim,
            self.max_degree, self.threshold, 
            self.std, self.mu_std
        )
        return data
    
def gaussian(x, mean, std):
    norm_const = 1 / (torch.sqrt(2 * torch.tensor(torch.pi)) * std)
    exp_value = torch.exp(-0.5 * (((x - mean) / std) ** 2))
    return norm_const * exp_value

class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        """
        K - the number of Gaussian Basis kernels
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.gamma = nn.Embedding(bond_types_number, 1, padding_idx=0)
        self.beta = nn.Embedding(bond_types_number, 1, padding_idx=0)

        # nn.init.uniform_(self.means.weight, 0, 3)
        # nn.init.uniform_(self.stds.weight, 0, 3)
        # nn.init.constant_(self.bias.weight, 0)
        # nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        """
        If n is the number of the atoms in a particular molecule:
        x - tensor of pairwise distances between the atoms with size [n, n], 
        edge_types - integer tensor of pairwise bond types between the atoms with size [n, n]
        """
        gamma = self.gamma(edge_types).squeeze(-1)            # [n, n, 1] -> [n, n]
        beta = self.beta(edge_types).squeeze(-1)              # [n, n, 1] -> [n, n]
        x = gamma * x + beta                                  # [n, n]
        means = self.means.weight.float().view(-1)            # [1, K] -> [K]
        stds = self.stds.weight.float().view(-1).abs() + 1e-2 # [1, K] -> [K]
        psi = gaussian(x[..., None], means, stds)             # [n, n, K]
        return psi
    
class NonLinear(nn.Sequential):
    def __init__(self, input_size, output_size):
        super().__init__(
            nn.Linear(input_size, input_size, bias=False), 
            nn.GELU(),
            nn.Linear(input_size, output_size, bias=False)
        )

class BondEncoding3D(nn.Module):
    """
    Compute Phi3D
    """
    def __init__(self, n_kernels=128, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kernels = n_kernels
        self.gaussian_kernels = GaussianLayer(self.n_kernels)
        self.perceptron = NonLinear(self.n_kernels, n_heads)

    def forward(self, pos, edge_types):
        """
        If n is the number of the atoms in a particular molecule:
        pos - tensor of the atom's positions with size [n, 3]
        edge_types - integer tensor of the atom's pairwise connection types with size [n, n]
        """
        x = torch.cdist(pos, pos)                     # [n, 3] -> [n, n]
        psi_3d = self.gaussian_kernels(x, edge_types) # [n, n] -> [n, n, n_kernels]
        phi_3d = self.perceptron(psi_3d)              # [n, n, n_kernels] -> [n, n, n_heads]
        phi_3d = phi_3d.permute(2, 0, 1)
        return phi_3d, psi_3d
    
def shortest_path_sequence(path, atoms_number, max_dist, atoms_types):
    bonds = np.zeros((atoms_number, atoms_number, max_dist), dtype=np.int64)
    atoms_to = np.broadcast_to(np.arange(atoms_number), (atoms_number, atoms_number))
    atoms_from = atoms_to.T
    atoms_inner = np.ones((atoms_number, atoms_number), dtype=np.int64)
    mask = path != -9999
    for k in range(1, max_dist+1):
        atoms_inner[mask] = path[atoms_from[mask], atoms_to[mask]]
        bonds[:, :, k-1][mask] = bond_types[
            atoms_types[atoms_to[mask]],
            atoms_types[atoms_inner[mask]]
        ]
        mask *= atoms_inner != atoms_from
        atoms_to = atoms_inner.copy()
        atoms_inner[...] = 0
    return bonds.T

def shortest_path(atoms, atoms_from_to):
    atoms_number = len(atoms)
    atoms_from, atoms_to = atoms_from_to
    row = atoms_from.numpy() 
    col = atoms_to.numpy() 
    is_bond = torch.ones_like(atoms_from).numpy()
    graph = csr_matrix((is_bond, (row, col)), shape=(atoms_number, atoms_number))
    dist_matrix, path = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    spatial_pos = torch.from_numpy((dist_matrix)).long()
    max_dist = np.amax(dist_matrix).astype(np.int64)
    return spatial_pos, path, max_dist

def shortest_path_distance(atoms, atoms_from_to):
    atoms_number = len(atoms)
    spatial_pos, path, max_dist = shortest_path(atoms, atoms_from_to)
    edge_input = shortest_path_sequence(path, atoms_number, max_dist, atoms)
    edge_input = torch.from_numpy(edge_input).long()
    return spatial_pos, edge_input, max_dist

# def preprocess_item(data):
#     edge_attr, edge_index, x = data.edge_attr, data.edge_index.to(torch.int64), data.x
#     N = x.size(0)
#     x = convert_to_single_emb(x)

#     # node adj matrix [N, N] bool
#     adj = torch.zeros([N, N], dtype=torch.bool)
#     adj[edge_index[0, :], edge_index[1, :]] = True

#     # edge feature here
#     if len(edge_attr.size()) == 1:
#         edge_attr = edge_attr[:, None]
#     attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
#     attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr) + 1
#     shortest_path_result, path = algos.floyd_warshall(adj.numpy())

#     max_dist = np.amax(shortest_path_result)
#     edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

#     spatial_pos = torch.from_numpy((shortest_path_result)).long()
#     attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

#     # combine
#     item.x = x
#     item.attn_bias = attn_bias
#     item.attn_edge_type = attn_edge_type
#     item.spatial_pos = spatial_pos
#     item.in_degree = adj.long().sum(dim=1).view(-1)
#     item.out_degree = item.in_degree # for undirected graph
#     item.edge_input = torch.from_numpy(edge_input).long()

#     return item

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / np.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class BondEncoding2D(nn.Module):
    """
    Compute PhiSTD, PhiEdge
    """
    def __init__(self, num_heads, num_spatial):
        super().__init__()
        self.num_heads = num_heads
        self.spd_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(bond_types_number, num_heads, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(num_spatial * num_heads * num_heads, 1)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atoms, atoms_from_to):
        # spatial_pos [n, n]
        # edge_input [n, n, max_dist]
        spatial_pos, edge_input, max_dist = shortest_path_distance(atoms, atoms_from_to)
        n = len(atoms)
        # [n, n] -> [n, n, n_heads] -> [n_heads, n, n]
        phi_spd = self.spd_encoder(spatial_pos).permute(2, 0, 1)
        # [n, n, max_dist] -> [n, n, max_dist, n_heads] 
        edge_input = self.edge_encoder(edge_input)
        # [n, n, max_dist, n_heads] -> [max_dist, n, n, n_heads] -> [max_dist, n x n, n_heads]
        edge_input_flat = edge_input.permute(2, 0, 1, 3).reshape(max_dist, -1, self.num_heads)
        # [num_spatial x n_heads x n_heads, 1] -> [num_spatial, n_heads, n_heads] -> [max_dist, n_heads, n_heads]
        edge_weights = self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist]
        # # [max_dist, n x n, n_heads] & [max_dist, n_heads, n_heads] -> [max_dist, n x n, n_heads]
        edge_input_flat = torch.bmm(edge_input_flat, edge_weights)
        # [max_dist, n x n, n_heads] -> [max_dist, n, n, n_heads] -> [n, n, max_dist, n_heads]
        edge_input = edge_input_flat.reshape(max_dist, n, n, self.num_heads).permute(1, 2, 0, 3)
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1
        # [n, n] - > [n, n, 1]
        spatial_pos_ = spatial_pos_.float().unsqueeze(-1)
        # [n, n, n_heads] / [n, n, 1] -> [n_heads, n, n]
        phi_edge = (edge_input.sum(-2) / spatial_pos_).permute(2, 0, 1)
        return phi_spd, phi_edge
    
class AtomEncoding2D(nn.Module):
    """
    Compute PsiDegree
    """
    def __init__(self, num_heads, num_degree, atom_feature_dim):
        super().__init__()
        self.num_heads = num_heads
        self.atom_feature_dim = atom_feature_dim
        self.atom_encoder = nn.Embedding(atom_types_number, atom_feature_dim * num_heads, padding_idx=0)
        self.degree_encoder = nn.Embedding(num_degree, atom_feature_dim * num_heads, padding_idx=0)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atoms, degrees):
        # atoms - [n]
        # degrees - [n]
        psi_atoms = self.atom_encoder(atoms)      # [n, n_heads x d]
        psi_degree = self.degree_encoder(degrees) # [n, n_heads x d]
        psi_degree = (psi_degree + psi_atoms).reshape(-1, self.num_heads, self.atom_feature_dim) # [n, n_heads, d]
        psi_degree = psi_degree.permute(1, 0, 2)
        return psi_degree
    
class AtomEncoding3D(nn.Module):
    """
    Compute PsiSum3DDistance
    """
    def __init__(self, num_heads, n_kernels, atom_feature_dim):
        super().__init__()
        self.num_heads = num_heads
        self.atom_feature_dim = atom_feature_dim
        self.W_3d = nn.Linear(n_kernels, num_heads * atom_feature_dim, bias=False)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, psi_3d):
        # psi_3d - [n, n, n_kernels]
        # [n, n, n_kernels] -> [n, n_kernels] -> [n, n_heads x d]
        phi_3d_sum = self.W_3d(psi_3d.sum(-2))
        # [n, n_heads x d] -> [n, n_heads, d]
        phi_3d_sum = phi_3d_sum.reshape(-1, self.num_heads, self.atom_feature_dim)
        # [n, n_heads, d] -> [n_heads, n, d]
        phi_3d_sum = phi_3d_sum.permute(1, 0, 2)
        return phi_3d_sum

class Dropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class AttentionBlock(nn.Module):
    def __init__(self, atom_feature_dim, scaling):
        super().__init__()
        self.scaling = scaling
        self.Q = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.K = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.V = nn.Linear(atom_feature_dim, atom_feature_dim, bias=False)
        self.dropout_module = Dropout(0.1, module_name=self.__class__.__name__)

    def forward(self, x, phi_degree, phi_3d_sum, phi_3d, phi_spd, phi_edge, delta_pos=None):
        """
        x - [n, d]
        phi_degree, phi_3d_sum, phi_3d, phi_spd, phi_edge - [n, n]
        """
        x = x + phi_degree + phi_3d_sum
        Q = self.Q(x) # [n, d]
        K = self.K(x) # [n, d]
        V = self.V(x) # [n, d]
        attn = Q @ K.transpose(-1, -2) * self.scaling # [n, n] # / np.sqrt(Q.size(-1)))
        attn = attn + phi_spd + phi_edge + phi_3d # [n, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_module(attn)#.view(self.num_heads, n, n)
        # attn = attn.unsqueeze(-1) * delta_pos.unsqueeze(1)
        # attn = attn.permute(0, 1, 4, 2, 3)
        attn = attn @ V # [n, d]
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, atom_feature_dim, num_heads):
        super().__init__()
        scaling = (atom_feature_dim // num_heads) ** -0.5
        self.heads = nn.Sequential(*[
            AttentionBlock(atom_feature_dim, scaling) for i in range(num_heads)
        ])
        self.W = nn.Linear(num_heads * atom_feature_dim, 1, bias=False)
        # self.force_proj1 = nn.Linear(embed_dim, 1)
        # self.force_proj2 = nn.Linear(embed_dim, 1)
        # self.force_proj3 = nn.Linear(embed_dim, 1)

    def forward(self, x, phi_degree, phi_3d_sum, phi_3d, phi_spd, phi_edge, delta_pos=None):
        """
        x - [n, d]
        phi_degree, phi_3d_sum, phi_3d, phi_spd, phi_edge - [n_heads, n, n]
        """
        # delta_pos - нормализованные радиус-вектора попарных расстояний между атомами [n, n, 3]
        attn = [
            head(x, phi_degree[h], phi_3d_sum[h], phi_3d[h], phi_spd[h], phi_edge[h], delta_pos) 
            for h, head in enumerate(self.heads)
        ]
        attn = torch.cat(attn, dim=-1)
        z = self.W(attn)
        # # f1 = self.force_proj1(x[:, :, 0, :]).view(n, 1)
        # # f2 = self.force_proj2(x[:, :, 1, :]).view(n, 1)
        # # f3 = self.force_proj3(x[:, :, 2, :]).view(n, 1)
        # # z = torch.cat([f1, f2, f3], dim=-1).float()
        # z = self.W(x)
        return z
    