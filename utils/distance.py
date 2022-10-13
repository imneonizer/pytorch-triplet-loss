class CosineDistance:
    """
    Serializable distance metric for VPTree
    """
    def __init__(self):
        from numpy import dot, sqrt
        self.dot = dot
        self.sqrt = sqrt

    def __call__(self, a, b):
        return 1 - self.dot(a, b) / (self.sqrt(self.dot(a,a)) * self.sqrt(self.dot(b,b)))