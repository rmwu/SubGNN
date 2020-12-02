import random

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Typing
from typing import List

class SubgraphDataset(Dataset):
    '''
    Stores subgraphs and their associated labels as well as precomputed similarities and border sets for the subgraphs
    '''

    def __init__(self, subgraph_list: List, cc_ids, N_border, NP_sim, I_S_sim, B_S_sim):
        # subgraph ids
        self.subgraph_list = subgraph_list
        self.cc_ids = cc_ids

        # precomputed border set
        self.N_border = N_border

        # precomputed similarity matrices
        self.NP_sim = NP_sim
        self.I_S_sim = I_S_sim
        self.B_S_sim = B_S_sim

    def __len__(self):
        '''
        Returns number of subgraphs
        '''
        return len(self.subgraph_list)

    def __getitem__(self, idx):
        '''
        Returns a single example from the datasest
        '''

        subgraph_ids = torch.LongTensor(self.subgraph_list[idx]) # list of node IDs in subgraph

        cc_ids = self.cc_ids[idx]
        N_border = self.N_border[idx] if self.N_border != None else None
        NP_sim = self.NP_sim[idx] if self.NP_sim != None else None
        I_S_sim = self.I_S_sim[idx] if self.I_S_sim != None else None
        B_S_sim = self.B_S_sim[idx] if self.B_S_sim != None else None

        idx = torch.LongTensor([idx])

        return (subgraph_ids, cc_ids, N_border, NP_sim, I_S_sim, B_S_sim, idx)


class SubgraphPairDataset(Dataset):
    '''
    Stores subgraph pairs and their associated labels as well as precomputed similarities and border sets for the subgraphs
    '''

    def __init__(self, subgraphs, labels, drugs, diseases, sample_negative=True):
        # instance of SubgraphDataset
        self.subgraphs = subgraphs
        # [(graph1_a, graph1_b, label1), ...] where graph1_a, graph1_b are ints
        self.labels = labels
        self.drugs = drugs
        self.diseases = diseases
        # this flag is False for test
        self.sample_negative = sample_negative

    def __len__(self):
        '''
        Returns number of subgraph pairs
        '''
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns a single example from the datasest
        '''
        # sample positive or negative
        # labels are all positive
        if self.sample_negative:
            label = int(random.random() > 0.5)
            if label:
                g1, g2, _ = self.labels[idx]
            else:
                # randomly sample
                g1 = random.sample(self.drugs, 1)[0]
                g2 = random.sample(self.diseases, 1)[0]
        # negatives are given
        # TODO
        else:
            g1, g2, label = self.labels[idx]

        return self.subgraphs[g1], self.subgraphs[g2], label

