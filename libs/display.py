#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from IPython import display


class Printer(object):

    def simple(self, obj):
        return display.display(display.Markdown(obj))

    def ul(self, obj):
        fmt = "\n".join(["- {}".format(e) for e in obj])
        return self.simple(fmt)

    def ol(self, obj):
        fmt = "\n".join(["{}. {}".format(idx+1, e) for idx, e in enumerate(obj)])
        return self.simple(fmt)

    def __call__(self, obj):
        if  isinstance(obj, (set, frozenset)):
            return self.ul(obj)
        elif  isinstance(obj, (list, tuple, np.ndarray)):
            return self.ol(obj)
        return self.simple(str)

    def bf(self, obj):
        return self("** {} **".format(obj))

    def tt(self, obj):
        return self("`{}`".format(obj))

d = Printer()
