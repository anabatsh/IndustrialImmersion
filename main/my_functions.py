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

# Number of all possible atom types
atom_types_number = 10
# Number of all possible bond types
bond_types_number = atom_types_number * (atom_types_number + 1) // 2 + 1

def integer_symmetric_matrix(v, h, w):
    """
    Generate a matrix of bond types' embeddings
    params: v, h, w - int
    """
    m = torch.randint(v, size=(h, w))
    m = torch.tril(m, 0) + torch.tril(m, -1).T
    return m

bond_types = integer_symmetric_matrix(bond_types_number, atom_types_number, atom_types_number)

def generate_atoms_types(n_atoms):
    """
    Generate random types for n atoms
    """
    return torch.randint(atom_types_number, size=(n_atoms,))

def generate_atoms_features(n_atoms, atom_feature_dim):
    """
    Generate random features for n atoms
    """
    return torch.randn(n_atoms, atom_feature_dim)

def generate_atoms_positions(n_atoms, std=1.0, mu_std=0.0):
    """
    Generate random positions for n atoms
    """
    pos = [torch.rand(3)]
    for _ in range(n_atoms - 1):
        pos.append((torch.rand(3) - 0.5) * 2 * std + pos[-1])
    pos = torch.vstack(pos)
    pos -= pos.mean(0)
    pos += torch.randn(3) * mu_std
    return pos

def generate_bonds(pos_from, pos_to, atoms_types_from, atoms_types_to, max_degree, threshold):
    """
    Find bonds for given atoms
    """
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

def generate_complex(n_protein_atoms, n_ligand_atoms, atom_feature_dim, 
                     max_degree, threshold, std=1.0, mu_std=1.0):
    """
    Generate a random protein-ligand complex
    """
    data = HeteroData()
    
    max_dist = np.sqrt(3) * std
    if threshold <= max_dist:
        lost_bonds = 1 - threshold / max_dist
        print(f'More than {100 * lost_bonds:.0f}% of the bonds could have been lost.')
        print('Increasing the threshold is recommended.')

    data['protein'].atoms = generate_atoms_types(n_protein_atoms)
    data['protein'].x = generate_atoms_features(n_protein_atoms, atom_feature_dim)
    data['protein'].pos = generate_atoms_positions(n_protein_atoms, std)

    data['ligand'].atoms = generate_atoms_types(n_ligand_atoms)
    data['ligand'].x = generate_atoms_features(n_ligand_atoms, atom_feature_dim)
    data['ligand'].pos = generate_atoms_positions(n_ligand_atoms, std, mu_std=mu_std)

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

def show_complex(data, field_name='x', ligand_name='ligand', protein_name='protein', 
                 ligand_bond_name='-', protein_bond_name='-', protein_ligand_bond_name='-',
                 with_labels=True, ax=None):
    """
    Display a given protein-ligand complex in the form of networkx graph
    """
    protein_nodes = range(data[protein_name][field_name].shape[0])
    ligand_nodes = range(data[ligand_name][field_name].shape[0])

    protein_protein_edges = data[protein_name, protein_bond_name, protein_name].edge_index.numpy().T
    ligand_ligand_edges = data[ligand_name, ligand_bond_name, ligand_name].edge_index.numpy().T
    protein_ligand_edges = data[protein_name, protein_ligand_bond_name, ligand_name].edge_index.numpy().T

    G = nx.Graph()

    for node in protein_nodes:
        G.add_node(f'{node}'+r'$_p$', label=node+1, color='pink')
    for edge in protein_protein_edges:
        edge = [f'{node}'+r'$_p$' for node in edge]
        G.add_edge(*edge, color='pink')

    for node in ligand_nodes:
        G.add_node(f'{node}'+r'$_l$', label=node+1, color='lightskyblue')
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
        
def show_complex_batch(batch, field_name='x', ligand_name='ligand', protein_name='protein', 
                       ligand_bond_name='-', protein_bond_name='-', protein_ligand_bond_name='-',
                       with_labels=True):
    """
    Display a batch of protein-ligand complexes in the form of networkx graph
    """
    n_5 = int(np.ceil(len(batch) / 5))
    _, axes = plt.subplots(n_5, 5, figsize=(15, 3*n_5))
    for data, ax in zip(batch, axes):
        show_complex(data, field_name, ligand_name, protein_name, 
                             ligand_bond_name, protein_bond_name, protein_ligand_bond_name, 
                             with_labels, ax=ax)
        ax.axis('on')
    plt.show()

class ProteinLigandComplexes(Dataset):
    """
    Dataset of protein-ligand complexes
    """
    def __init__(self, N=1000, n_protein_atoms_max=20, 
                 n_ligand_atoms_max=10, atom_feature_dim=8, max_degree=5, 
                 threshold=1.8, std=1.0, mu_std=2.0):
        self.N = N
        self.n_protein_atoms_max = n_protein_atoms_max
        self.n_ligand_atoms_max = n_ligand_atoms_max
        self.atom_feature_dim = atom_feature_dim
        self.max_degree = max_degree
        self.threshold = threshold
        self.std = std
        self.mu_std = mu_std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        n_protein_atoms = torch.randint(1, self.n_protein_atoms_max, (1,))
        n_ligand_atoms = torch.randint(1, self.n_ligand_atoms_max, (1,))
        data = generate_complex(
            n_protein_atoms, n_ligand_atoms, 
            self.atom_feature_dim,
            self.max_degree, self.threshold, 
            self.std, self.mu_std
        )
        return data
    
def gaussian(x, mean, std):
    """
    Calculate a value of a gaussian function at x
    """
    norm_const = 1 / (torch.sqrt(2 * torch.tensor(torch.pi)) * std)
    exp_value = torch.exp(-0.5 * (((x - mean) / std) ** 2))
    return norm_const * exp_value

class GaussianLayer(nn.Module):
    """
    Gaussian Basis Kernel functions
    """
    def __init__(self, n_kernels=128):
        """
        n_kernels - the number of Gaussian Basis kernels (K)
        """
        super().__init__()
        self.means = nn.Embedding(1, n_kernels)
        self.stds = nn.Embedding(1, n_kernels)
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
    def __init__(self, n_heads, n_kernels):
        super().__init__()
        self.gaussian_kernels = GaussianLayer(n_kernels)
        self.perceptron = NonLinear(n_kernels, n_heads)

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
    
def shortest_path_sequence(path, n_atoms, max_dist, atoms_types):
    """
    Find sequences of the atoms in the shortest paths
    """
    bonds = np.zeros((n_atoms, n_atoms, max_dist), dtype=np.int64)
    atoms_to = np.broadcast_to(np.arange(n_atoms), (n_atoms, n_atoms))
    atoms_from = atoms_to.T
    atoms_inner = np.ones((n_atoms, n_atoms), dtype=np.int64)
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
    """
    Compute the shortest paths
    """
    n_atoms = len(atoms)
    atoms_from, atoms_to = atoms_from_to
    row = atoms_from.numpy() 
    col = atoms_to.numpy() 
    is_bond = torch.ones_like(atoms_from).numpy()
    graph = csr_matrix((is_bond, (row, col)), shape=(n_atoms, n_atoms))
    dist_matrix, path = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
    spatial_pos = torch.from_numpy((dist_matrix)).long()
    max_dist = np.amax(dist_matrix).astype(np.int64)
    return spatial_pos, path, max_dist

def shortest_path_distance(atoms, atoms_from_to):
    """
    Compute the shortest paths and the coresponding sequences of the atoms
    """
    n_atoms = len(atoms)
    spatial_pos, path, max_dist = shortest_path(atoms, atoms_from_to)
    edge_input = shortest_path_sequence(path, n_atoms, max_dist, atoms)
    edge_input = torch.from_numpy(edge_input).long()
    return spatial_pos, edge_input, max_dist

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
    def __init__(self, n_heads, n_spatial):
        super().__init__()
        self.n_heads = n_heads
        self.spd_encoder = nn.Embedding(n_spatial, n_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(bond_types_number, n_heads, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(n_spatial * n_heads * n_heads, 1)
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
        edge_input_flat = edge_input.permute(2, 0, 1, 3).reshape(max_dist, -1, self.n_heads)
        # [num_spatial x n_heads x n_heads, 1] -> [num_spatial, n_heads, n_heads] -> [max_dist, n_heads, n_heads]
        edge_weights = self.edge_dis_encoder.weight.reshape(-1, self.n_heads, self.n_heads)[:max_dist]
        # # [max_dist, n x n, n_heads] & [max_dist, n_heads, n_heads] -> [max_dist, n x n, n_heads]
        edge_input_flat = torch.bmm(edge_input_flat, edge_weights)
        # [max_dist, n x n, n_heads] -> [max_dist, n, n, n_heads] -> [n, n, max_dist, n_heads]
        edge_input = edge_input_flat.reshape(max_dist, n, n, self.n_heads).permute(1, 2, 0, 3)
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
    def __init__(self, max_degree, atom_feature_dim):
        super().__init__()
        # self.n_heads = n_heads
        self.atom_feature_dim = atom_feature_dim
        self.atom_encoder = nn.Embedding(atom_types_number, atom_feature_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(max_degree, atom_feature_dim, padding_idx=0)
        # self.atom_encoder = nn.Embedding(atom_types_number, atom_feature_dim * n_heads, padding_idx=0)
        # self.degree_encoder = nn.Embedding(max_degree, atom_feature_dim * n_heads, padding_idx=0)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atoms, degrees):
        # atoms - [n]
        # degrees - [n]
        psi_atoms = self.atom_encoder(atoms)      # [n, d]
        psi_degree = self.degree_encoder(degrees) # [n, d]
        psi_degree = psi_degree + psi_atoms
        # psi_degree = (psi_degree + psi_atoms).reshape(-1, self.n_heads, self.atom_feature_dim) # [n, n_heads, d]
        # psi_degree = psi_degree.permute(1, 0, 2)
        return psi_degree
    
class AtomEncoding3D(nn.Module):
    """
    Compute PsiSum3DDistance
    """
    def __init__(self, n_kernels, atom_feature_dim):
        super().__init__()
        # self.n_heads = n_heads
        self.atom_feature_dim = atom_feature_dim
        self.W_3d = nn.Linear(n_kernels, atom_feature_dim, bias=False)
        # self.W_3d = nn.Linear(n_kernels, n_heads * atom_feature_dim, bias=False)
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, psi_3d):
        # psi_3d - [n, n, n_kernels]
        # [n, n, n_kernels] -> [n, n_kernels] -> [n, d]
        phi_3d_sum = self.W_3d(psi_3d.sum(-2))
        # phi_3d_sum = phi_3d_sum.reshape(-1, self.n_heads, self.atom_feature_dim)
        # [n, n_heads, d] -> [n_heads, n, d]
        # phi_3d_sum = phi_3d_sum.permute(1, 0, 2)
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

    def forward(self, x, phi_3d, phi_spd, phi_edge, delta_pos):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n, n]
        """
        Q = self.Q(x) # [n + 1, d]
        K = self.K(x) # [n + 1, d]
        V = self.V(x) # [n + 1, d]
        attn = Q @ K.transpose(-1, -2) * self.scaling # [n + 1, n + 1] # / np.sqrt(Q.size(-1)))
        stride = x.shape[0] - phi_3d.shape[0]
        attn[stride:, stride:] += phi_spd + phi_edge + phi_3d # [n + 1, n + 1]
        attn = F.softmax(attn, dim=-1) # [n + 1]
        attn = self.dropout_module(attn)
        # attn = attn.unsqueeze(-1) * delta_pos.unsqueeze(1)
        attn = attn @ V # [n + 1, d]
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, atom_feature_dim, model_dim=1):
        super().__init__()
        scaling = (atom_feature_dim // n_heads) ** -0.5
        self.heads = nn.Sequential(*[
            AttentionBlock(atom_feature_dim, scaling) for i in range(n_heads)
        ])
        self.W = nn.Linear(n_heads * atom_feature_dim, model_dim, bias=False)

    def forward(self, x, phi_3d, phi_spd, phi_edge, delta_pos=None):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n_heads, n, n]
        """
        # delta_pos - нормализованные радиус-вектора попарных расстояний между атомами [n, n, 3]
        attn = [
            head(x, phi_3d[h], phi_spd[h], phi_edge[h], delta_pos) # [n + 1, d]
            for h, head in enumerate(self.heads)
        ]
        attn = torch.cat(attn, dim=-1) # [n + 1, n_heads x d]
        attn = self.W(attn)            # [n + 1, d]
        return attn
    
class TransformerMLayer(nn.Module):
    def __init__(self, n_heads, atom_feature_dim, model_dim):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(n_heads, atom_feature_dim, model_dim)
        self.feedforward = NonLinear(model_dim, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, x, phi_3d, phi_spd, phi_edge):
        """
        x - [n + 1, d]
        phi_3d, phi_spd, phi_edge - [n_heads, n, d]
        """
        attn = self.multihead_attn(x, phi_3d, phi_spd, phi_edge)
        x = self.norm1(x + attn)
        z = self.feedforward(x)
        x = self.norm2(x + z)
        return x # [n + 1, d]

class PositionalEncoding(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim):
        super().__init__()
        self.atom_encoder_2d = AtomEncoding2D(max_degree=max_degree, atom_feature_dim=atom_feature_dim)
        self.atom_encoder_3d = AtomEncoding3D(n_kernels=n_kernels, atom_feature_dim=atom_feature_dim)
        self.bond_encoder_2d = BondEncoding2D(n_heads=n_heads, n_spatial=n_spatial)
        self.bond_encoder_3d = BondEncoding3D(n_heads=n_heads, n_kernels=n_kernels)

    def forward(self, data_atoms, data_bonds):
        """
        data_atoms.x - atoms features
        data_atoms.pos - atoms positions
        data_atoms.atoms - atoms types
        data_bonds.edge_index - bonds: [atoms from, atoms_to]
        data_bonds.edge_attr - [bond length, bonds types]
        """
        n = len(data_atoms.atoms)
        edge_types = torch.zeros(n, n).long()
        atoms_from, atoms_to = data_bonds.edge_index
        edge_types[atoms_from, atoms_to] = edge_types[atoms_to, atoms_from] = data_bonds.edge_attr[1]
        degrees = (edge_types != 0).sum(1)

        phi_3d, psi_3d = self.bond_encoder_3d(data_atoms.pos, edge_types)
        phi_spd, phi_edge = self.bond_encoder_2d(data_atoms.atoms, data_bonds.edge_index)
        phi_degree = self.atom_encoder_2d(data_atoms.atoms, degrees)
        phi_3d_sum = self.atom_encoder_3d(psi_3d)

        return {
            'atoms': (phi_degree, phi_3d_sum),   # [n, d]
            'bonds': (phi_3d, phi_spd, phi_edge) # [n_heads, n, n]
        }

class TransformerMEncoder(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim, n_encoder_layers):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, atom_feature_dim))
        self.positional_encoder = PositionalEncoding(
            n_heads=n_heads, max_degree=max_degree, n_kernels=n_kernels, 
            n_spatial=n_spatial, atom_feature_dim=atom_feature_dim
        )
        self.transformer_m_layers = nn.Sequential(*[
            TransformerMLayer(
                n_heads=n_heads, atom_feature_dim=atom_feature_dim, model_dim=atom_feature_dim
            ) for _ in range(n_encoder_layers)
        ])

    def forward(self, data_atoms, data_bonds):
        phi = self.positional_encoder(data_atoms, data_bonds)
        phi_degree, phi_3d_sum = phi['atoms']
        phi_3d, phi_spd, phi_edge = phi['bonds']

        x = data_atoms.x + phi_degree + phi_3d_sum # [n, d]
        x = torch.cat([self.cls_token, x]) # [n + 1, d]

        for layer in self.transformer_m_layers:
            x = layer(x, phi_3d, phi_spd, phi_edge)
        return x[0] # cls_token

class TransformerM(nn.Module):
    def __init__(self, n_heads, max_degree, n_kernels, n_spatial, atom_feature_dim, n_encoder_layers):
        super().__init__()
        self.encoder = TransformerMEncoder(
            n_heads=n_heads, max_degree=max_degree, n_kernels=n_kernels, 
            n_spatial=n_spatial, atom_feature_dim=atom_feature_dim, 
            n_encoder_layers=n_encoder_layers
        )
        self.regressor = nn.Linear(atom_feature_dim, 1)
        
    def forward(self, data_atoms, data_bonds):
        out = self.encoder(data_atoms, data_bonds)
        out = self.regressor(out)
        return out