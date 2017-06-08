class DecisionNode:
    """
    Represents a node / leaf in a tree.
    :param col:     Column of the variable to split in this node
    :param value:   Value of the variable used to split
    :param result:  Result contained in leaf nodes
    :param tb:      Left branch of the split
    :param fb:      Right branch of the split
    """

    def __init__(self, col=-1, value=None, result=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.result = result
        self.tb = tb
        self.fb = fb
