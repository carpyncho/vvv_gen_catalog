from IPython import display as d


class Printer(object):
    
    def simple(self, obj):
        return d.display(d.Markdown(obj))
    
    def __call__(self, obj):
        if isinstance(obj, str):
            return self.simple(str)
        elif  isinstance(obj, (list, tuple, set, frozenset)):
            return self.iterable(obj)
        
    def bf(self, obj):
        self("** {} **".format(obj))
        
    def tt(self, obj):
        self("`{} **".format(obj))
        