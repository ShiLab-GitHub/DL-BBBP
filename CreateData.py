import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from utils import *
import torch

# 读取数据集文件
train_set_name = 'data/train.csv.balance' #y_test_indices.csv
df_train = pd.read_csv(train_set_name)
test_set_name = 'data/test.csv'
df_test = pd.read_csv(test_set_name)
df = pd.concat([df_train,df_test])
# 移除重复的smiles
compound_iso_smiles = set(list( df['smi'] ))
max_len = max([len(i) for i in compound_iso_smiles])
print('Max SMILES length =', max_len, '.')
# 383 in train.csv.balance, 400 in y_test_indices.csv

smiles_graph = {}
smiles_MACCS = {}
flag = 0 # 无法转换成graph的smiles直接跳过，flag记录跳过的数量

for smiles in compound_iso_smiles:
    mol = Chem.MolFromSmiles(smiles)
    # graph
    g = smiles_to_graph(smiles)
    if g is not None:
        smiles_graph[smiles] = g
    else:
        flag += 1
        continue
    # MACCS
    # 去除MACCS中的结构信息，mask表示需要保留的位数。
    # MACCS每位含义：https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
    mask = [1,2,3,4,5,6,7,9,10,12,18,20,27,29,35,42,44,46,49,56,88,103,134,159,161,164]
    fp = MACCSkeys.GenMACCSKeys(mol) # <rdkit.DataStructs.cDataStructs.ExplicitBitVect object>
    fp = torch.tensor([np.float32(_) for _ in fp.ToBitString()[1:]]) # 166位
    smiles_MACCS[smiles] = torch.tensor([fp[m-1] for m in mask])

processed_data_file = 'data/processed/dataset.pt'
if not os.path.isfile(processed_data_file):
    # read file
    df = pd.read_csv(train_set_name)
    train_drugs, train_label = list(df['smi']),list(df['label'])
    train_drugs, train_label = np.asarray(train_drugs), np.asarray(train_label)
    df = pd.read_csv(test_set_name)
    test_drugs,  test_label  = list(df['smi']),list(df['label'])
    test_drugs,  test_label  = np.asarray(test_drugs), np.asarray(test_label)

    dataset_drugs = np.hstack((train_drugs, test_drugs))
    dataset_label = np.hstack((train_label, test_label))
    print('preparing dataset.pt')
    dataset_data = TestbedDataset(root='data', dataset='dataset',
                            smi=dataset_drugs, label=dataset_label,
                            smiles_graph=smiles_graph, smiles_MACCS=smiles_MACCS)
else:
    print(processed_data_file, ' are already created.')

print('Removed smiles number =', flag, '.')