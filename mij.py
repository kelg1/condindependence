class Mij: 

    def __init__(self, gamma=0):
        self.gamma = gamma
        self.fit_ = False 

    def fit(self, y, X):
        y1, y2 = y.T[0], y.T[1]
        Fn1 = CondMargin(h_x=1e-1, h_y=1e-3)
        Fn2 = CondMargin(h_x=1e-1, h_y=1e-3)
        Fn1.fit(y1, X)
        Fn2.fit(y2, X)
        self.Fn1 = Fn1
        self.Fn2 = Fn2
        #R1 = self.Fn1.rank(y1, X)
        #R2 = self.Fn2.rank(y2, X)
        R1 = self.Fn1.rank(y1, X)
        R2 = self.Fn2.rank(y2, X)
        self.R1 = R1
        self.R2 = R2
        self.fit_ = True


    def evaluate(self, i, j):
        if not self.fit_ :
            raise NotImplementedErrory
        assert(len(self.R1)==len(self.R2))
        n = len(self.R1)
        Ri1 = self.R1[i]
        Ri2 = self.R2[i]
        Rj1 = self.R1[j]
        Rj2 = self.R2[j]
        mij = (1 - np.maximum(Ri1, Rj1)/n) * (1 - np.maximum(Ri2, Rj2)/n) - \
                (.5 - .5*((Ri1)/n)**2)*(.5 - .5*((Ri2)/n)**2) - \
                (.5 - .5*((Rj1)/n)**2)*(.5 - .5*((Rj2)/n)**2) + 1./9
#        mij = (1 - np.maximum(Ri1, Rj1)) * (1 - np.maximum(Ri2, Rj2)) - \
#                (.5 - .5*((Ri1))**2)*(.5 - .5*((Ri2))**2) - \
#                (.5 - .5*((Rj1))**2)*(.5 - .5*((Rj2))**2) + 1./9
        return mij
    
    @staticmethod
    def positive(x):
        return np.maximum(x, 0.)