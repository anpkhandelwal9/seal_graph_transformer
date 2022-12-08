from ogb.utils import smiles2graph

# if you use Pytorch Geometric (requires torch_geometric to be installed)
from ogb.lsc import PygPCQM4Mv2Dataset
pyg_dataset = PygPCQM4Mv2Dataset(root = './dataset/pcqm', smiles2graph = smiles2graph)
print(pyg_dataset[0])