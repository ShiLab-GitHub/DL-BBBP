import os
import numpy as np
from sklearn import metrics
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from rdkit import Chem
import networkx as nx

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    if c_size == 1:
        return 
    return c_size, features, edge_index

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='dataset', # root='data', dataset=.pt文件名
                 transform=None,pre_transform=None,
                 smi=None,label=None,
                 smiles_graph=None,smiles_MACCS=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]): # 'data\\processed\\dataset.pt'
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(smi,label,smiles_graph,smiles_MACCS)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, smi, labels, smiles_graph, smiles_MACCS):
        assert (len(smi) == len(labels)), "SMILES list and label list must be the same length!"
        data_list = []
        data_len = len(smi)
        pos = 0
        neg = 0
        for i in range(data_len):
            smiles = smi[i]
            label = labels[i]
            if label:
                pos += 1
            else:
                neg += 1
            if smiles in smiles_graph:
                c_size, features, edge_index = smiles_graph[smiles]
                if c_size == 1:
                    continue
                GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([label]))
                GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
                GCNData.__setitem__('smiles', smiles)
                GCNData.__setitem__('MACCS', smiles_MACCS[smiles])
                data_list.append(GCNData)
        np.random.shuffle(data_list)
        print('label 1: ', str(pos))
        print('label 0: ', str(neg))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(self.dataset, 'file constructed. Saving.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
  
def statistic(y,f): # labels,predicts
    if len(y) != len(f):
        raise ValueError("预测值和标签的长度不相等")
    tp = 0 # true positive
    tn = 0 # true negative
    fp = 0 # false positive
    fn = 0 # false negative
    threshold = 0.5
    for i in range(len(y)):
        if y[i] == 1 and f[i] >= threshold:
            tp += 1
        if y[i] == 0 and f[i] < threshold:
            tn += 1
        if y[i] == 1 and f[i] < threshold:
            fn += 1
        if y[i] == 0 and f[i] >= threshold:
            fp += 1
    
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return sn,sp,acc,mcc

def acc(y,f): # labels,predicts
    if len(y) != len(f):
        raise ValueError("预测值和标签的长度不相等")
    threshold = 0.5
    g = np.where(f >= threshold, 1, 0)
    tmp = (y+g)%2
    tmp = tmp.sum()
    acc = tmp / len(y)
    acc = 1-acc
    return acc

def auc(y,f): # labels,predicts
    if len(y) != len(f):
        raise ValueError("预测值和标签的长度不相等")
    fpr, tpr, thresholds = metrics.roc_curve(y, f, pos_label=1)
    return metrics.auc(fpr, tpr)