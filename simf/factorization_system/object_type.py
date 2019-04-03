class ObjectType(object):
    def __init__(self, name=None, rank=10):
        self.name = name if name else str(id(self))
        self.rank = rank
        self.length = None

    def __str__(self):
        return str(self.name)

    def set_rank(self, r):
        self.rank = r

    def get_shape(self):
        return self.length, self.rank

    def get_name(self):
        return self.name

    def get_rank(self):
        return self.rank
