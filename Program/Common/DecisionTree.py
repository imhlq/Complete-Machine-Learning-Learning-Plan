# Decision Tree v0.1
# xhou.me

import numpy as np
import pandas as pd

class Node:
    def __init__(self, cond):
        self.children = []
        self.cond = cond    # for attr value(parent class)
        self.data = None    # for attr class
        self.data_name = None
        self.isLeaf = False

    def setLeaf(self, data):
        # set this node as leaf node
        self.data = data
        self.isLeaf = True
    
    def addChild(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

    def __str__(self, level=0):
        if self.isLeaf:
            ret = "\t"*level+repr((self.cond, '*',self.data))+"\n"
        else:
            ret = "\t"*level+repr((self.cond, self.data_name))+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret


class DecisionTree:
    def __init__(self):
        self.root = Node('root')
    
    def calc_Entropy(self, pi):
        classDict = np.unique(pi)
        Ent = 0
        for k in classDict:
            p_k = np.shape(pi[pi==k])[0] / np.shape(pi)[0]
            Ent += - p_k * np.log2(p_k)
        return Ent

    def calc_Gini(self, pi):
        # return gini of pi
        classDict = np.unique(pi)
        Gini = 0
        for k in classDict:
            p_k = np.shape(pi[pi==k])[0] / np.shape(pi)[0]
            Gini += p_k * p_k
        Gini = 1 - Gini
        return Gini

    def purity(self, xi, yi, method='IG'):
        # method: the purity concept used to calculate
        # Default:IG-Information Gain, GR - Gain Ratio, GI - Gini Index
        if method == 'IG':
            # Information Gain
            # - Calculate Entropy of total
            Ent = self.calc_Entropy(yi)
            
            # Calculate Gain
            Gain = []
            # - Calculate Entropy of every attribute
            for ai in range(np.shape(xi)[1]):
                # loop every attribute class
                Ent_v = 0
                classDict = np.unique(xi[:, ai])
                for ai_a in classDict:
                    # loop every attribute
                    yi_a = yi[xi[:, ai] == ai_a]
                    Ent_i = len(yi_a) / len(yi) * self.calc_Entropy(yi_a)
                    Ent_v += Ent_i
                # index is important!
                Gain.append(Ent - Ent_v)
            # return the best one
            attr = np.argmax(Gain)
            return int(attr)
        
        elif method == 'GR':
            # Gain ratio
            # - Calculate Entropy of total
            Ent = self.calc_Entropy(yi)
            
            # Calculate Gain
            Gain = []
            gIV = []
            # - Calculate Entropy of every attribute
            for ai in range(np.shape(xi)[1]):
                # loop every attribute class
                Ent_v = 0
                gIV_v = 0
                classDict = np.unique(xi[:, ai])
                for ai_a in classDict:
                    # loop every attribute
                    yi_a = yi[xi[:, ai] == ai_a]
                    Ent_i = len(yi_a) / len(yi) * self.calc_Entropy(yi_a)
                    gIV_i = len(yi_a) / len(yi) * np.log2(len(yi_a) / len(yi))
                    Ent_v += Ent_i
                    gIV_v += gIV_i
                # index is important!
                Gain.append(Ent - Ent_v)
                gIV.append(gIV_v)
            gain_ratio = np.array(Gain) / np.array(gIV)
            # Heuristic Select by 1. Gain above average, 2. Highest gain_ratio
            step1 = np.argwhere(Gain > np.mean(Gain))
            step2 = max(step1, key=lambda a: np.array(Gain)[a])
            return int(step2)
    
        elif method == 'GI':
            # Gain ratio
            # - Calculate Entropy of total
            Gini = self.calc_Gini(yi)
            
            # Calculate GI
            Gi = []
            # - Calculate Entropy of every attribute
            for ai in range(np.shape(xi)[1]):
                # loop every attribute class
                Gi_v = 0
                classDict = np.unique(xi[:, ai])
                for ai_a in classDict:
                    # loop every attribute
                    yi_a = yi[xi[:, ai] == ai_a]
                    Gi_i = len(yi_a) / len(yi) * self.calc_Gini(yi_a)
                    Gi_v += Gi_i
                # index is important!
                Gi.append(Gi_v)

            attr = np.argmin(Gi)
            return int(attr)



    def createBranch(self, xi, yi, root_node=None):
        # -- Recursive --
        # 1. If yi same, return
        # 2. Choose xi
        # 3. Create Node
        if all(yi==yi[0]):
            root_node.setLeaf(int(yi[0])) # int target
            return 

        if root_node is None:
            root_node = self.root

        if xi.shape[1] == 1:
            # Last attr
            attr = 0
        else:
            attr = self.purity(xi, yi, 'GI')
        
        # Create branch
        root_node.data = attr
        root_node.data_name = ['Color', 'Root', 'Knocks', 'Texture', 'Umbilicus', 'Touch'][attr]
        attr_class = np.unique(xi[:, attr])
        for ev_attr in attr_class:

            new_Node = Node(ev_attr)
            root_node.addChild(new_Node)
            # Recursive
            new_xi = xi[xi[:,attr] == ev_attr]
            new_yi = yi[xi[:,attr] == ev_attr]

            if new_xi.size == 0:
                # set leaf node, with most common class now
                counts = np.bincount(yi)
                new_Node.setLeaf(int(yi[np.argmax(counts)]))

            self.createBranch(new_xi[:, np.arange(np.shape(xi)[1]) != attr], new_yi, new_Node)
        return

    def pruning(self):
        # postpruning
        pass

    def __str__(self):
        return str(self.root)

def createData(filename, columns, label):
    # waterlemon 3a
    dt = pd.read_csv(filename)
    xi = np.array(dt[columns])
    yi = np.array(dt[label])
    return xi, yi

if __name__ == "__main__":
    DataCol = ['Color', 'Root', 'Knocks', 'Texture', 'Umbilicus', 'Touch']
    TargetCol = ['Label']
    xi, yi = createData('../Homework@imhlq/Chap3/watermelon3_0_En.csv', DataCol, TargetCol)

    dt = DecisionTree()
    dt.createBranch(xi, yi)
    print(dt)
    #print(dt.display())