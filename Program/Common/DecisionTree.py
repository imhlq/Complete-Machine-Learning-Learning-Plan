# Decision Tree v0.1
# xhou.me

import numpy as np

class Node:
    def __init__(self, name):
        self.children = []
        self.name = name    # for attr value(parent class)
        self.data = None    # for attr class
    
    def addChild(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

    def __str__(self):
        nstr = '<Node:%s|%s>|Children:%d>\n' % (self.name, self.data, len(self.children))
        return nstr


class DecisionTree:
    def __init__(self):
        self.root = Node('root')
    
    def entropy(self, pi):
        classDict = np.unique(pi)
        Ent = 0
        for k in classDict:
            p_k = np.shape(pi[pi==k])[0]
            Ent += - p_k * np.log2(p_k)
        return Ent

    def purity(self, xi, yi, method='IG'):
        # method: the purity concept used to calculate
        # Default:IG-Information Gain, GR - Gain Ratio, GI - Gini Index
        if method == 'IG':
            # Information Gain
            # - Calculate Entropy of total
            Ent = self.entropy(yi)
            
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
                    Ent_i = len(yi_a) / len(yi) * self.entropy(yi_a)
                    Ent_v += Ent_i
                # index is important!
                Gain.append(Ent - Ent_v)
            # return the best one
            attr = np.argmax(Gain)
            return attr

    def createBranch(self, xi, yi, root_node=None):
        # -- Recursive --
        # 1. If yi same, return
        # 2. Choose xi
        # 3. Create Node
        if len(yi)==0 or all(yi==yi[0]):
            return root_node
        
        if root_node is None:
            root_node = self.root

        attr = self.purity(xi, yi, 'IG')
        # Create branch
        root_node.data = attr
        attr_class = np.unique(xi[:, attr])
        for ev_attr in attr_class:
            new_Node = Node(ev_attr)
            root_node.addChild(new_Node)
            # Recursive
            self.createBranch(xi[xi[:,attr] == ev_attr], yi[xi[:,attr] == ev_attr], new_Node)

    def display(self, node=None, lev=0):
        # how to do this?
        out = '-' * lev
        if node is None:
            node = self.root

        out += str(node)
        for child in self.root.children:
            out += self.display(child, lev+1)

        return out

if __name__ == "__main__":
    dt = DecisionTree()
    dt.createBranch(xi, yi)
    print(dt.display())