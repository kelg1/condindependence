from datagen import *
# from conditionalmargin import *
# from mij import *
# from sampler import *
# from statistic import * 
# from nadaraya import *
import sys 
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy 
from scipy import spatial 

class Statistic_T:
    def __init__(self, ):
        pass
    def evaluate(self, R1, R2, K):
        n = np.max(R1) 
        R1 = R1.reshape(-1,1) / n
        R2 = R2.reshape(-1,1) / n
        a = np.maximum(R1, R1.T)
        b = np.maximum(R2, R2.T)
        c = - .25*(1*(1-R1**2)*(1-R2**2)) - .25*(1*(1-R1**2)*(1-R2**2)).T + 1./9
        mij = (1-a)*(1-b) + c
        res = np.multiply(mij, K).sum()/(n**2)
        return res

class RankHat:
    def __init__(self, kx=10):
        self.fit_ = False
        self.kx = kx
    def fit(self, X, Y1):
        n = len(X)
        neigh = NearestNeighbors(n_neighbors=self.kx)
        neigh.fit(X)
        self.A  = (neigh.kneighbors_graph(X, mode='connectivity') - np.eye(n))/(self.kx-1)
        Y1 = Y1.reshape(-1,1)
        self.Ui = np.ravel(np.dot(self.A, (Y1.T - Y1 < 0)).sum(0)/n)
        self.rank = self.Ui.argsort().argsort()
        self.K = np.exp(-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X,
                                                                                        metric='euclidean')**2))
        self.fit_ = True
        

class USamplerUnderNull:
    def __init__(self, ):
        pass
    
    def fit(self, X, U1, U2):
        self.X  = X
        self.U1 = U1
        self.U2 = U2
        self.size = len(U1)
        
    def sample(self, kx):
        index = np.random.choice(np.arange(self.size), size=self.size, replace=True)
        X0 = self.X[index]
        neigh = NearestNeighbors(n_neighbors=kx)
        neigh.fit(self.X)
        neighbors = neigh.kneighbors_graph(X0, mode='connectivity')
        U10 = self.U1[index]
        U20 = np.zeros_like(self.U2)
        for i, p_s in enumerate(neighbors.toarray()):
            U20[i] = np.random.choice(self.U2, p=p_s/kx)
        return X0, U10, U20
    
class TestCopulas:
    def __init_(self):
        pass
     
    def bootstrap_test(self, X, Y1, Y2, B, alpha, kx):
        R1 = RankHat(kx)
        R2 = RankHat(kx)

        R1.fit(X, Y1)
        R2.fit(X, Y2)

        stat = Statistic_T()
        eval_stat_on_dataset = stat.evaluate(R1.rank, R2.rank, R1.K)

        self.dico = {'T': eval_stat_on_dataset, 'T0':[]}
        eval_under_null = []
        for b in range(B):
            sampler_ = USamplerUnderNull()
            sampler_.fit(X, R1.Ui, R2.Ui)
            X0, U01, U02 = sampler_.sample(kx)
            K0 = np.exp(-scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X0,
                                                                                        metric='euclidean')**2))
            R10, R20 = U01.argsort().argsort() , U02.argsort().argsort()
            stat0 = Statistic_T()
            eval_under_null.append(stat0.evaluate(R10, R20, K0))
            
        self.dico['T0'] = eval_under_null


def compute_t1error(dico, alpha):
    t1_vs_param = {}
    for param, v in dico.items():
        reject_under_true_null = 0
        for exp, dico_test in enumerate(v):  
            reject_under_true_null += (dico_test['T'] >= np.percentile(dico_test['T0'], 100 - alpha))
        t1_vs_param[param] = reject_under_true_null / len(dico_test)
    return t1_vs_param


def compute_t2error(dico, alpha):
    t2_vs_param = {}
    for param, v in dico.items():
        accept_under_false_null = 0
        for exp, dico_test in enumerate(v):  
            accept_under_false_null += (dico_test['T']< np.percentile(dico_test['T0'], 100 - alpha))
        t2_vs_param[param] = accept_under_false_null / len(dico_test)
    return t2_vs_param


kx = 230 #'already set by cross val procedure'
n=1000
d=1
# #sys.stdout.write('approx. process time: ?')
np.random.seed(1)
test_dico_exp2 = defaultdict(list)
for a in [.4]:
    for nexp in range(10):
        sys.stdout.write('\r exp to compute type II error rate: {}, a: {}, n: {}, d: {}'.format(nexp, a, n, d))
        X, Y1, Y2, beta = sampling_linear_model(n, d, a)
        test_cop = TestCopulas()
        test_cop.bootstrap_test(X, Y1, Y2, 100, .1, kx)
        test_dico_exp2[a].append(test_cop.dico)

#sys.stdout.write('approx. process time: ?')
test_dico_exp1 = defaultdict(list)
kx=230 #set by cross val procedure
for a in [0]:
    for nexp in range(15):
        sys.stdout.write('\r exp to compute type I error rate: {}, a: {}, n: {}, d: {}'.format(nexp, a, n, d))
        X, Y1, Y2, beta = sampling_linear_model(n, d, a)
        test_cop = TestCopulas()
        test_cop.bootstrap_test(X, Y1, Y2, 100, .1, kx)
        test_dico_exp1[a].append(test_cop.dico)

#print('\n error_type_2_rate', compute_t2error(test_dico_exp2, 5))
print('\n error_type_1_rate', compute_t1error(test_dico_exp1, 5))


# def exp_type1():
# 	stat_on_datasetH1 = defaultdict(list)
# 	stat_under_bootstrap = defaultdict(list)
# 	test0 = defaultdict(list)
# 	np.random.seed()
# 	for a in tqdm.tqdm(np.linspace(0.01, .3, 10)):
# 		sys.stdout.write('\r a = {:.1f}'.format(a))
# 		###sampling from linear model
# 		X, Y1, Y2, beta = sampling_linear_model(n, d, 0)  

# 		stat = TStatistic(.1, gamma=0, h_x=1, h_y=0)
# 		ta = stat.evaluate(1, np.column_stack((Y1, Y2)), X)
# 		stat_on_datasetH1[a].append(ta)

# 		for bstrap in range(100):
# 		    ### Data sampling
# 		    condni = CondIndepSampler(sigma_X=.1, kernel='RBF')
# 		    condni.fit(X, np.column_stack((Y1,Y2)))
# 		    new_dataset = condni.sample(n)
# 		    Xnew, Ynew = new_dataset[:,0:1], new_dataset[:,1:]
# 		    stat = TStatistic(.1, gamma=0, h_x=1, h_y=0)
# 		    stat_under_bootstrap[a].append(stat.evaluate(1, Ynew, Xnew))

# 		test0[a] = 1*(stat_on_datasetH1[a] >= np.percentile(stat_under_bootstrap[a], q))
# 	return test0


# # Type II error

# def exp_type2():
# 	stat_on_datasetH1 = defaultdict(list)
# 	stat_under_bootstrap = defaultdict(list)
# 	test1 = defaultdict(list)
# 	np.random.seed()
# 	for a in tqdm.tqdm(np.linspace(0.01, .3, 10)):
# 		sys.stdout.write('\r a = {:.1f}'.format(a))
# 		X, Y1, Y2, beta = sampling_linear_model(n, d, 2*a)  

# 		stat = TStatistic(a*4, gamma=0, h_x=1, h_y=0)
# 		#stat.fit(X, Y1,Y2)
# 		ta = stat.evaluate(1, np.column_stack((Y1, Y2)), X)
# 		stat_on_datasetH1[a].append(ta)
# 		#sys.stdout.write('\r The test statistic on the dataset Tn={:.2f}'.format(ta))

# 		for bstrap in range(100):
# 		    ### Data sampling
# 		    condni = CondIndepSampler(sigma_X=a*2, kernel='RBF')
# 		    condni.fit(X, np.column_stack((Y1,Y2)))
# 		    new_dataset = condni.sample(n)
# 		    Xnew, Ynew = new_dataset[:,0:1], new_dataset[:,1:]
# 		    stat = TStatistic(a*4, gamma=0, h_x=1, h_y=0)
# 		    #stat.fit(Xnew, *Ynew)
# 		    stat_under_bootstrap[a].append(stat.evaluate(1, Ynew, Xnew))

# 		test1[a] = 1*(stat_on_datasetH1[a] >= np.percentile(stat_under_bootstrap[a], 95))
# 	return test1
# type1 = defaultdict(list)
# type2 = defaultdict(list)

# n_exp = 100
# sys.stdout.write("""The running time is about 1 hour.
# 	The results have already been saved in './results.mat'.
# 	""")
# for exp in tqdm.tqdm(range(n_exp)):
# 	t1 = exp_type1()
# 	t2 = exp_type2()
# 	type1.update({exp: t1})
# 	type2.update({exp: t2})

# error1 = defaultdict(list)
# error2 = defaultdict(list)
# for exp, v in type1.items():
# 	for a, t in v.items():
# 		error1[a].append(t[0][0])


# for exp, v in type2.items():
# 	for a, t in v.items():
# 		error2[a].append(1-t[0][0])


# print('type I error', {k: np.mean(v)/2. for k,v in error1.items()})
# print('type II error', {k: np.mean(v)/2. for k,v in error2.items()})

# ## plot the results 
# results = np.load('./results.npy')
# f = plt.figure()
# ax = f.add_subplot(111)
# ax.plot(np.linspace(0.01, .3, 10), results[:,0], label='typeIerror', c='r')
# ax.set_xlabel('a')
# ax.set_ylabel('typeIerror',color='r')
# ax=ax.twinx()
# ax.plot(np.linspace(0.01, .3, 10), results[:,1],label='typeIIerror',c='b')
# ax.set_ylabel('typeIIerror',color='b')
# plt.show()





