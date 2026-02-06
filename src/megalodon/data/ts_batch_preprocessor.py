import torch

from megalodon.data.random_rotations import random_rotations
from torch_geometric.utils import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean


def make_graph_fully_connected(edge_index, edge_attr, batch):
    # Load bond information from the dataloader
    bond_edge_index, bond_edge_attr = sort_edge_index(
        edge_index=edge_index, edge_attr=edge_attr, sort_by_row=False
    )
    bond_edge_index, bond_edge_attr = coalesce(bond_edge_index, bond_edge_attr,
                                               reduce="min")

    # Create Fully Connected Graph instead
    edge_index_global = (
        torch.eq(batch.unsqueeze(0),
                 batch.unsqueeze(-1)).int().fill_diagonal_(0)
    )
    edge_index_global, _ = dense_to_sparse(edge_index_global)
    edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
    # Handle cases where edge_attr is N or NxK
    if edge_attr.dim() == 1:  # Case where edge_attr is (N,)
        edge_attr_tmp = torch.full(
            size=(edge_index_global.size(-1),),
            fill_value=0,
            device=edge_index_global.device,
            dtype=torch.long,
        )
    else:  # Case where edge_attr is (N, K)
        K = edge_attr.size(1)
        edge_attr_tmp = torch.full(
            size=(edge_index_global.size(-1), K),
            fill_value=0,
            device=edge_index_global.device,
            dtype=torch.long,
        )
    edge_index_global = torch.cat([edge_index_global, bond_edge_index], dim=-1)
    edge_attr_tmp = torch.cat([edge_attr_tmp, bond_edge_attr], dim=0)

    edge_index_global, edge_attr_global = coalesce(edge_index_global, edge_attr_tmp,
                                                   reduce="max")

    edge_index_global, edge_attr_global = sort_edge_index(
        edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
    )
    return edge_index_global, edge_attr_global



class TsBatchPreProcessor:
    def __init__(self, 
                 aug_rotations=False,
                 scale_coords=1.0,
                 ts_ratio=1.0):
        """
        Args:
            aug_rotations: Apply random rotations
            scale_coords: Scale coordinates
            ts_ratio: Ratio of TS vs product/reactant training
                      0.0 = only product/reactant (easy, has clear stereo)
                      1.0 = only TS (normal, harder)
                      0.2 = 20% TS, 80% product/reactant
        """
        self.aug_rotations = aug_rotations
        self.scale_coords = scale_coords
        self.ts_ratio = ts_ratio

    def __call__(self, batch):
        """Custom collate function to apply augmentations and curriculum learning."""
        batch_size = torch.max(batch.batch) + 1
        if self.aug_rotations and batch.pos is not None:
            rotations = random_rotations(batch_size, batch.pos.dtype, batch.pos.device)
            rotations = rotations[batch.batch]
            batch.pos = torch.bmm(rotations, batch.pos.unsqueeze(-1)).squeeze(-1)

        if self.scale_coords and batch.pos is not None:
            batch.pos = batch.pos / self.scale_coords

        # Curriculum learning: replace TS with product/reactant based on ratio
        if self.ts_ratio < 1.0:
            # For each molecule, decide if it should be TS or product/reactant
            use_ts = torch.rand(batch_size, device=batch.batch.device) < self.ts_ratio
            # For non-TS molecules, decide if product or reactant (ONE choice per molecule)
            use_product = torch.rand(batch_size, device=batch.batch.device) > 0.5
            
            for mol_idx in range(batch_size):
                if not use_ts[mol_idx]:
                    mol_mask = batch.batch == mol_idx
                    
                    if use_product[mol_idx]:
                        # Use product coordinates
                        batch.ts_coord[mol_mask] = batch.p_coord[mol_mask]
                    else:
                        # Use reactant coordinates
                        batch.ts_coord[mol_mask] = batch.r_coord[mol_mask]

        batch['edge_index'], batch['edge_attr'] = make_graph_fully_connected(edge_index=batch["edge_index"],
                                                                             edge_attr=batch["edge_attr"],
                                                                             batch=batch["batch"])
        
        # Replace edge_attr channels for product/reactant molecules (using SAME choice)
        if self.ts_ratio < 1.0:
            edge_batch = batch.batch[batch["edge_index"][0]]
            for mol_idx in range(batch_size):
                if not use_ts[mol_idx]:
                    edge_mol_mask = edge_batch == mol_idx
                    
                    if use_product[mol_idx]:
                        # Both channels = product topology
                        batch["edge_attr"][edge_mol_mask, 0] = batch["edge_attr"][edge_mol_mask, 1]
                    else:
                        # Both channels = reactant topology  
                        batch["edge_attr"][edge_mol_mask, 1] = batch["edge_attr"][edge_mol_mask, 0]
        
        batch['ts_coord'] = (
                batch['ts_coord'] -
                scatter_mean(batch['ts_coord'], index=batch.batch, dim=0, dim_size=batch_size)[
                    batch.batch]
        ).float()

        batch['bmat_r'] = batch["edge_attr"][..., 0].long()
        batch['bmat_p'] = batch["edge_attr"][..., 1].long()
        batch["numbers"] = batch["numbers"].long()
        if "charges" in batch.keys():
            batch["charges"] = batch["charges"].long()
        return batch