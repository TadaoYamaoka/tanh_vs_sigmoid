import chainer
import chainer.functions as F
import chainer.links as L

# Network definition
class MyNet(chainer.Chain):

    def __init__(self):
        super(MyNet, self).__init__(
            l1=L.Linear(None, 32),
            l2=L.Linear(None, 32),
            l3=L.Linear(None, 1),
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)
