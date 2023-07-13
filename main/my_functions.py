import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


atom_types_number = 10
bond_types_number = atom_types_number ** 2 + 1

def integer_symmetric_matrix(v, h, w):
    m = torch.randint(v, size=(h, w))
    m = torch.tril(m, 0) + torch.tril(m, -1).T
    return m

bond_types = integer_symmetric_matrix(bond_types_number, atom_types_number, atom_types_number)

def generate_atoms(atoms_number):
    return torch.randint(atom_types_number, size=(atoms_number,))

def generate_positions(atoms_number, std=1.0, mu_std=0.0):
    pos = [torch.rand(3)]
    for _ in range(atoms_number - 1):
        pos.append((torch.rand(3) - 0.5) * 2 * std + pos[-1])
    pos = torch.vstack(pos)
    pos -= pos.mean(0)
    pos += torch.randn(3) * mu_std
    return pos

def generate_connections(pos_from, pos_to, atoms_types_from, atoms_types_to, max_degree, threshold):
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
    max_degree, threshold,
    std=1.0, mu_std=0.0
):
    data = HeteroData()
    
    data['protein'].atoms = generate_atoms(protein_atoms_number)
    data['protein'].pos = generate_positions(protein_atoms_number, std)

    data['ligand'].atoms = generate_atoms(ligand_atoms_number)
    data['ligand'].pos = generate_positions(ligand_atoms_number, std, mu_std=mu_std)

    # protein-protein
    protein_protein, dists, bonds = generate_connections(
        data['protein'].pos, data['protein'].pos,
        data['protein'].atoms, data['protein'].atoms,
        max_degree, threshold
    )
    data['protein', '-', 'protein'].edge_index = protein_protein
    data['protein', '-', 'protein'].edge_attr = (dists, bonds)

    # ligand-ligand
    ligand_ligand, dists, bonds = generate_connections(
        data['ligand'].pos, data['ligand'].pos, 
        data['ligand'].atoms, data['ligand'].atoms, 
        max_degree, threshold
    )
    data['ligand', '-', 'ligand'].edge_index = ligand_ligand
    data['ligand', '-', 'ligand'].edge_attr = (dists, bonds)
    
    # protein-ligand
    protein_ligand, dists, bonds = generate_connections(
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
        max_degree=5, threshold=1.8,
        std=1.0, mu_std=2.0
    ):
        self.N = N
        self.protein_atoms_number_max = protein_atoms_number_max
        self.ligand_atoms_number_max = ligand_atoms_number_max
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

class Encoding3D(nn.Module):
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
        x = torch.cdist(pos, pos)                  # [n, 3] -> [n, n]
        psi = self.gaussian_kernels(x, edge_types) # [n, n] -> [n, n, n_kernels]
        phi = self.perceptron(phi)                 # [n, n, n_kernels] -> [n, n, n_heads]
        return phi