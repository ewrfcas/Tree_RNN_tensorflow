import random


# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node:  # a node in the tree
    def __init__(self, label, id=400, word=None):
        self.label = label
        self.id = id
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class Tree:

    def __init__(self, treeString, openChar='(', closeChar=')'):
        self.open = openChar
        self.close = closeChar
        self.id_count = 0
        treeString = treeString.split(' ')
        self.index = int(treeString[-1])
        treeString = self.add_blank(treeString[0:-1])
        tokens = treeString.split(' ')
        self.root = self.parse(tokens)
        self.set_id(self.root)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def set_id(self,root):
        if root is None:
            return
        else:
            if root.isLeaf:
                root.id=self.id_count
                self.id_count+=1
            else:
                self.set_id(root.left)
                self.set_id(root.right)

    def add_blank(self, treeString):
        treeString=' '.join(treeString)
        treeString=treeString.replace('(','( ').strip()
        treeString=treeString.replace(')',' )').strip()
        treeString=treeString.replace('  ',' ')
        return treeString.strip()

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(float(tokens[1]))

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = tokens[2]
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def clearFprop(node, words):
    node.fprop = False


def loadTrees(file,data_type):
    with open(file, 'r') as fid:
        trees=[]
        datas=fid.readlines()
        for i,l in enumerate(datas):
            print("\r[loading "+data_type+" data:%d/%d]" % (i+1,len(datas)), end='      ', flush=True)
            trees.append(Tree(l))
        print('\n')
    return trees

def binarize_labels(trees):
    def binarize_node(node, _):
        if node.label < 2:
            node.label = 0
        elif node.label > 2:
            node.label = 1

    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)
