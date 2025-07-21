import numpy as np
import torch
import torch.nn as nn


class Graph_Joint():
    def __init__(self, layout='mocha', strategy='distance', max_hop=2, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'mixamo':
            self.num_node = 22
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1, 0, 1, 2,         # Spine  
                        3, 4,               # Neck
                        3, 6, 7, 8,         # LeftArm
                        3, 10, 11, 12,      # RightArm
                        0, 14, 15, 16,      # RightLeg
                        0, 18, 19, 20]      # LeftLeg

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'Xia':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                 # Hip
                        0,  1,  2,  3,     # Left Leg
                        0,  5,  6,  7,     # Right Leg
                        0,  9,             # Spine
                        10, 11,            # Neck
                        10, 13, 14, 15,    # Left Arm
                        10, 17, 18, 19]    # Right Arm

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'ian':            
            self.num_node = 23
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                  # Hip
                        0,  1,  2,  3,      # Chest (spine)
                        4,  5,              # Neck
                        4,  7,  8,  9,      # Right Arm
                        4, 11, 12, 13,      # Left Arm
                        0, 15, 16, 17,      # Right Leg
                        0, 19, 20, 21]      # Left Leg

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'mocha':            
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                  # Hip
                        0,  1,  2,  3,      # Left Leg
                        0,  5,  6,  7,      # Spine
                        8,  9, 10, 11,      # Left Arm
                        8, 13, 14,          # Neck & Head
                        8, 16, 17, 18,      # Right Arm
                        0, 20, 21, 22]      # Right Leg

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'adult2child':            
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                  # Hip
                        0,  1,  2,  3,      # Spine
                        4,  5,  6,  7,      # Neck & Head
                        4,  9, 10, 11, 12, 12,  # Right Arm
                        4, 15, 16, 17, 18, 18,  # Left Arm
                        0, 21, 22, 23, 24, 25,  # Right Leg
                        0, 27, 28, 29, 30, 31]  # Left Leg

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'bandai':            
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            parents = [-1,                  # Hip
                        0,  1,              # Spine
                        2,  3,              # Neck
                        2,  5,  6,  7,      # Left Arm
                        2,  9, 10, 11,      # Right Arm
                        0, 13, 14, 15,      # Left Leg
                        0, 17, 18, 19]      # Right Leg

            neighbor_link = [(i, parents[i]) for i in range(len(parents))]
            neighbor_link = neighbor_link[1:]   # remove (0, -1)
            self.edge = self_link + neighbor_link
            self.center = 0
        
        else:
            assert layout=='mixamo' or layout=='Xia', "Wrong layout"

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A


class Graph_Bodypart():
    def __init__(self, layout='mocha', strategy='distance', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):        
        if layout=='mixamo':       # same topology on bodypart level
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # Neck     -> 1
            # LeftArm  -> 2
            # RightArm -> 3
            # RightLeg -> 4
            # LeftLeg  -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'Xia':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # LeftLeg     -> 1
            # LeftArm  -> 2
            # Neck -> 3
            # RightArm -> 4
            # RightLeg  -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'ian':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # LeftLeg     -> 1
            # LeftArm  -> 2
            # Neck -> 3
            # RightArm -> 4
            # RightLeg  -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'mocha':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # LeftLeg     -> 1
            # LeftArm  -> 2
            # Neck -> 3
            # RightArm -> 4
            # RightLeg  -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == 'adult2child':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # Neck -> 1
            # RightArm -> 2
            # LeftArm  -> 3
            # RightLeg -> 4
            # LeftLeg -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        elif layout == 'bandai':
            self.num_node = 6
            self_link = [(i, i) for i in range(self.num_node)]
            # Spine    -> 0
            # Neck     -> 1
            # LeftArm  -> 2
            # RightArm -> 3
            # LeftLeg  -> 4
            # RightLeg -> 5
            neighbor_link = [(0,1), (0,2), (0,3), (0,4), (0,5)]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        else:
            assert layout=='mixamo' or layout=='Xia', "Wrong layout"


    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class PoolJointToBodypart(nn.Module):
    def __init__(self, layout):
        super().__init__()
        if layout == 'mixamo':
            self.Spine = [0, 1, 2, 3]
            self.Neck = [4, 5]
            self.LeftArm = [6, 7, 8, 9]
            self.RightArm = [10, 11, 12, 13]
            self.RightLeg = [14, 15, 16, 17]
            self.LeftLeg = [18, 19, 20, 21]

            njoints = 22
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.Neck, 1] = 1.0
            weight[self.LeftArm, 2] = 1.0
            weight[self.RightArm, 3] = 1.0
            weight[self.RightLeg, 4] = 1.0
            weight[self.LeftLeg, 5] = 1.0
        
        elif layout == 'Xia':
            self.Spine = [0, 9, 10]
            self.LeftLeg = [1, 2, 3, 4]
            self.RightLeg = [5, 6, 7, 8]
            self.Neck = [11, 12]
            self.LeftArm = [13, 14, 15, 16]
            self.RightArm = [17, 18, 19, 20]
            
            njoints = 21
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.LeftLeg, 1] = 1.0
            weight[self.RightLeg, 2] = 1.0
            weight[self.Neck, 3] = 1.0
            weight[self.LeftArm, 4] = 1.0
            weight[self.RightArm, 5] = 1.0

        elif layout == 'Xia2':
            self.Spine = [0, 9, 10]
            self.LeftLeg = [0, 1, 2, 3, 4]
            self.RightLeg = [0, 5, 6, 7, 8]
            self.Neck = [10, 11, 12]
            self.LeftArm = [10, 13, 14, 15, 16]
            self.RightArm = [10, 17, 18, 19, 20]
            
            njoints = 21
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.LeftLeg, 1] = 1.0
            weight[self.RightLeg, 2] = 1.0
            weight[self.Neck, 3] = 1.0
            weight[self.LeftArm, 4] = 1.0
            weight[self.RightArm, 5] = 1.0
        
        elif layout == 'ian':
            self.Spine = [0, 1, 2, 3, 4]
            self.Neck = [5, 6]
            self.RightArm = [7, 8, 9, 10]
            self.LefttArm = [11, 12, 13, 14]
            self.RightLeg = [15, 16, 17, 18]
            self.LeftLeg = [19, 20, 21, 22]

            njoints = 23
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.LeftLeg, 1] = 1.0
            weight[self.LefttArm, 2] = 1.0
            weight[self.Neck, 3] = 1.0
            weight[self.RightArm, 4] = 1.0
            weight[self.RightLeg, 5] = 1.0
        
        elif layout == 'mocha':
            self.Spine = [0, 5, 6, 7, 8]
            self.LeftLeg = [1, 2, 3, 4]
            self.LefttArm = [9, 10, 11, 12]
            self.Neck = [13, 14, 15]
            self.RightArm = [16, 17, 18, 19]
            self.RightLeg = [20, 21, 22, 23]

            njoints = 24
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.LeftLeg, 1] = 1.0
            weight[self.LefttArm, 2] = 1.0
            weight[self.Neck, 3] = 1.0
            weight[self.RightArm, 4] = 1.0
            weight[self.RightLeg, 5] = 1.0
        
        elif layout == 'adult2child':
            self.Spine = [0, 1, 2, 3, 4]
            self.Neck = [5, 6, 7, 8]
            self.RightArm = [9, 10, 11, 12, 13, 14]
            self.LefttArm = [15, 16, 17, 18, 19, 20]
            self.RightLeg = [21, 22, 23, 24, 25, 26]
            self.LeftLeg = [27, 28, 29, 30, 31, 32]

            njoints = 33
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.Neck, 1] = 1.0
            weight[self.RightArm, 2] = 1.0
            weight[self.LefttArm, 3] = 1.0
            weight[self.RightLeg, 4] = 1.0
            weight[self.LeftLeg, 5] = 1.0
        
        elif layout == 'bandai':
            self.Spine = [0, 1, 2]
            self.Neck = [3, 4]
            self.LefttArm = [5, 6, 7, 8]
            self.RightArm = [9, 10, 11, 12]
            self.LeftLeg = [13, 14, 15, 16]
            self.RightLeg = [17, 18, 19, 20]

            njoints = 21
            nbody = 6
            weight = torch.zeros(njoints, nbody, dtype=torch.float32, requires_grad=False)
            weight[self.Spine, 0] = 1.0
            weight[self.Neck, 1] = 1.0
            weight[self.LefttArm, 2] = 1.0
            weight[self.RightArm, 3] = 1.0
            weight[self.LeftLeg, 4] = 1.0
            weight[self.RightLeg, 5] = 1.0
            
        else:
            assert layout=='mixamo' or layout=='Xia' or layout=='Xia2' \
                    or layout=='ian' or layout=='mocha' or layout=='bandai', "Wrong layout"

        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


class UnpoolBodypartToJoint(nn.Module):
    def __init__(self, layout):
        super().__init__()
        if layout == 'mixamo':
            self.Spine = [0, 1, 2, 3]
            self.Neck = [4, 5]
            self.LeftArm = [6, 7, 8, 9]
            self.RightArm = [10, 11, 12, 13]
            self.RightLeg = [14, 15, 16, 17]
            self.LeftLeg = [18, 19, 20, 21]

            nbody = 6
            njoints = 22
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.Neck] = 1.0
            weight[2, self.LeftArm] = 1.0
            weight[3, self.RightArm] = 1.0
            weight[4, self.RightLeg] = 1.0
            weight[5, self.LeftLeg] = 1.0
        
        elif layout == 'Xia':
            self.Spine = [0, 9, 10]
            self.LeftLeg = [1, 2, 3, 4]
            self.RightLeg = [5, 6, 7, 8]
            self.Neck = [11, 12]
            self.LeftArm = [13, 14, 15, 16]
            self.RightArm = [17, 18, 19, 20]

            nbody = 6
            njoints = 21
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.LeftLeg] = 1.0
            weight[2, self.RightLeg] = 1.0
            weight[3, self.Neck] = 1.0
            weight[4, self.LeftArm] = 1.0
            weight[5, self.RightArm] = 1.0

        elif layout == 'Xia2':
            self.Spine = [0, 9, 10]
            self.LeftLeg = [0, 1, 2, 3, 4]
            self.RightLeg = [0, 5, 6, 7, 8]
            self.Neck = [10, 11, 12]
            self.LeftArm = [10, 13, 14, 15, 16]
            self.RightArm = [10, 17, 18, 19, 20]

            nbody = 6
            njoints = 21
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.LeftLeg] = 1.0
            weight[2, self.RightLeg] = 1.0
            weight[3, self.Neck] = 1.0
            weight[4, self.LeftArm] = 1.0
            weight[5, self.RightArm] = 1.0
        
        elif layout == 'ian':
            self.Spine = [0, 1, 2, 3, 4]
            self.Neck = [5, 6]
            self.RightArm = [7, 8, 9, 10]
            self.LefttArm = [11, 12, 13, 14]
            self.RightLeg = [15, 16, 17, 18]
            self.LeftLeg = [19, 20, 21, 22]

            nbody = 6
            njoints = 23
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.LeftLeg] = 1.0
            weight[2, self.LefttArm] = 1.0
            weight[3, self.Neck] = 1.0
            weight[4, self.RightArm] = 1.0
            weight[5, self.RightLeg] = 1.0
        
        elif layout == 'mocha':
            self.Spine = [0, 5, 6, 7, 8]
            self.LeftLeg = [1, 2, 3, 4]
            self.LefttArm = [9, 10, 11, 12]
            self.Neck = [13, 14, 15]
            self.RightArm = [16, 17, 18, 19]
            self.RightLeg = [20, 21, 22, 23]

            nbody = 6
            njoints = 24
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.LeftLeg] = 1.0
            weight[2, self.LefttArm] = 1.0
            weight[3, self.Neck] = 1.0
            weight[4, self.RightArm] = 1.0
            weight[5, self.RightLeg] = 1.0
        
        elif layout == 'adult2child':
            self.Spine = [0, 1, 2, 3, 4]
            self.Neck = [5, 6, 7, 8]
            self.RightArm = [9, 10, 11, 12, 13, 14]
            self.LefttArm = [15, 16, 17, 18, 19, 20]
            self.RightLeg = [21, 22, 23, 24, 25, 26]
            self.LeftLeg = [27, 28, 29, 30, 31, 32]

            nbody = 6
            njoints = 33
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.Neck] = 1.0
            weight[2, self.RightArm] = 1.0
            weight[3, self.LefttArm] = 1.0
            weight[4, self.RightLeg] = 1.0
            weight[5, self.LeftLeg] = 1.0
        
        elif layout == 'bandai':
            self.Spine = [0, 1, 2]
            self.Neck = [3, 4]
            self.LefttArm = [5, 6, 7, 8]
            self.RightArm = [9, 10, 11, 12]
            self.LeftLeg = [13, 14, 15, 16]
            self.RightLeg = [17, 18, 19, 20]

            nbody = 6
            njoints = 21
            weight = torch.zeros(nbody, njoints, dtype=torch.float32, requires_grad=False)
            weight[0, self.Spine] = 1.0
            weight[1, self.Neck] = 1.0
            weight[2, self.LefttArm] = 1.0
            weight[3, self.RightArm] = 1.0
            weight[4, self.LeftLeg] = 1.0
            weight[5, self.RightLeg] = 1.0

        else:
            assert layout=='mixamo' or layout=='Xia' \
                    or layout=='ian' or layout=='mocha' or layout=='bandai', "Wrong layout"


        scale = torch.sum(weight, axis=0, keepdim=True)
        weight = weight / scale
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.weight))        
        return x


if __name__ == '__main__':
    x_in = torch.randn(3, 15, 60, 21)
    pool_test = PoolJointToBodypart(layout='Xia2')
    upool_test = UnpoolBodypartToJoint(layout='Xia2')
    # print(pool_test.weight)
    # print(upool_test.weight)

    # print(PoolJointToBodypart()(x_in).shape)
    print(pool_test(x_in).shape)
    print(upool_test(pool_test(x_in)).shape)