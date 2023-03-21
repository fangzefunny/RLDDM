import os 

import pickle
import numpy as np 
import pandas as pd 
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.stats import uniform, beta
from scipy.optimize import minimize

from scipy.special import softmax
from utils.ddm import wfpt
from utils.viz import viz
viz.get_style()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))
eps_ = 1e-12
max_ = 1e12

# ------------------------ #
#       Preprocess         #  
# ------------------------ #

def preprocess(raw_data):

    # subject list 
    sub_lst = raw_data['subj_idx'].unique()
    
    name_dict = {
        's': 'stim',
        'c': 'act',
        'r': 'rew',
    }

    data = raw_data.rename(columns=name_dict)
    data['stim'] = data['stim'].apply(lambda x: x-1)
    data['act']  = data['act'].apply(lambda x: x-1)
    
    # split to blocks
    out_data = {}
    for subj_idx in sub_lst:
        sub_data = data.query(f'subj_idx=={subj_idx}')
        out_data[subj_idx] = sub_data 

    return out_data 

# ------------------------ #
#          Models          #  
# ------------------------ #

class simpleBuffer:

    def __init__(self):
        self.keys = ['s', 'a', 'r']
        self.reset()

    def push(self, m_dict):
        '''Add a sample trajectory'''
        for k in m_dict.keys():
            self.m[k] = m_dict[k]

    def sample(self, *args):
        '''Sample a trajectory'''
        lst = [self.m[k] for k in args]
        if len(lst) == 1: return lst[0]
        else: return lst 

    def reset(self):
        '''Empty the cached trajectory'''
        self.m = {k: 0 for k in self.keys}

class rlddm:
    name     = 'rlddm'
    bnds     = [(-20, 20), (0, 1), (1e-3, 20), (0, 1)]
    pbnds    = [( -5,  5), (0,.5), (1e-3, 3),  (0,.5)]
    p_name   = ['β', 'α', 'a', 't0']
    priors   = [uniform(-20, 20), beta(1.2, 1.2), uniform(1e-3, 20), uniform(0, 1)]
    n_params = len(bnds) 

    def __init__(self, nS, nA, params):
        self.nS = nS 
        self.nA = nA
        self.v  = 0
        self._init_Q()
        self._init_memory()
        self._load_params(params)

    def _init_Q(self):
        self.q_SA = np.zeros([self.nS, self.nA])

    def _init_memory(self):
        self.mem = simpleBuffer()

    def _load_params(self, params):
        self.beta  = params[0] # drift rate linear weight
        self.alpha = params[1] # learning rate 
        self.a     = params[2] # decision bound
        self.t0     = params[3] # non-decision time

    def policy(self, s):
        v = self.dirft_rate(s)
        p = 1/(np.exp(-self.a*v))
        return np.array([1-p, p])
    
    def rt_hat(self, s):
        v = self.dirft_rate(s)
        return (.5*self.a/v)*np.tanh(.5*self.a*v) + self.t0
    
    def drift_rate(self, s):
        return self.beta*(self.q_SA[s, 1] - self.q_SA[s, 0])
    
    def eval_rt(self, s, a, t):
        '''Evaluate the loglikehood of the time

        Using the Wiener first passage time distribution

        Args:
            s: stimulus
            a: action 
            t: the reaction time

        Output:
            log(p): the LLH of the time calculated 
                using the Wiener First Passage time distribution
        '''

        # rm init decision time 
        rt = np.max([eps_, t-self.t0])

        # pos: a=1, drift rate v =  beta[Q(a2) - Q(a1)] 
        # neg: a=0, drift rate v = -beta[Q(a2) - Q(a1)] 
        v = self.beta*(self.q_SA[s, 1] - self.q_SA[s, 0]) * 2*(a-.5)
        p = wfpt(rt, v, self.a)
        if (np.isnan(p).sum()) or (p.sum()==0): p=eps_
        
        return p
        
    def learn(self):
        
        # get data 
        s, a, r = self.mem.sample('s', 'a', 'r')
        # update 
        rpe = r - self.q_SA[s, a]
        self.q_SA[s, a] += self.alpha*rpe 

# ------------------------- #
#           Fit             # 
# ------------------------- #

class model:

    def __init__(self, agent):
        self.agent = agent 

    def fit(self, data, seed, method='map', verbose=False):
        
        # get the parameter bound
        bnds  = self.agent.bnds
        pbnds = self.agent.pbnds

        # init parameters
        rng = np.random.RandomState(seed)
        param0 = [pbnd[0] + (pbnd[1] - pbnd[0]
                    ) * rng.rand() for pbnd in pbnds]

        ## Fit the params 
        if verbose: print('init with params: ', param0) 
        res = minimize(self.loss_fn, param0, args=(data, method), 
                        bounds=bnds, options={'disp': verbose})
        if verbose: print(f'''  Fitted params: {res.x}, 
                    MLE loss: {res.fun}''')
        
        return res.x, res.fun 
    
    def loss_fn(self, params, data, method):
        
        # the log likelihood 
        obj = self.loglike(params, data)
        
        # the log prior 
        if method == 'map': 
            logprior = np.sum([np.max([prior.logpdf(param), -max_])
                        for prior, param in 
                        zip(self.agent.priors, params)])
            obj += logprior

        return -obj
    
    def loglike(self, params, block_data):

        # init the model 
        nS = len(block_data['stim'].unique())
        nA = 2
        subj = self.agent(nS, nA, params)

        # start fitting 
        loglike = 0 
        for _, row in block_data.iterrows():

            # obtain the inputs
            s = int(row['stim'])
            a = int(row['act'])
            r = row['rew']
            t = row['rt'] 

            # store to memory 
            mem = {'s': s, 'a': a, 'r': r}
            subj.mem.push(mem)

            # evaluate the decision time 
            like = subj.eval_rt(s, a, t)
            loglike += np.log(like+eps_)

            # learn 
            subj.learn()
        
        return loglike
    
    def sim(self, block_data, params):
        
        # init the model 
        nS = len(block_data['stim'].unique())
        nA = 2
        subj = self.agent(nS, nA, params)

        ## init a blank dataframe to store simulation
        col = ['acc', 'rt_mean', 'drif_rate']
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        for _, row in block_data.iterrows():

            # obtain the inputs
            s = row['stim']
            a = row['act']
            r = row['rew']
            t = row['rt'] 

            # store to memory 
            mem = {'s': s, 'a': a, 'r': r}
            subj.mem.push(mem)

            # evaluate the decision time 
            rt_mean = subj.rt_hat(s)
            pi = subj.policy(s)
            v  = subj.drift_rate(s)

            # record the vals 
            pred_data.loc[t, 'acc']        = pi[a]
            pred_data.loc[t, 'rt_mean']    = rt_mean
            pred_data.loc[t, 'dirft_rate'] = v   

            # learn 
            subj.learn()

        return pd.concat([block_data, pred_data], axis=1)
    

def fit_all_sub(agent, data):

    # get a list of subject   
    sub_lst = list(data.keys())

    # fit to each subject
    fit_info = {}
    for sub_id in tqdm(sub_lst):

        # fit the subj
        subj = model(agent)
        min_loss, opt_param = max_, False
        n_fit, seed = 5, 2023
        for i in range(n_fit):
            fit_param, fit_loss = subj.fit(
                data[sub_id], seed+i, method='map')
            if fit_loss < min_loss:
                min_loss  = fit_loss 
                opt_param = fit_param

        # save the parameters
        fit_info[sub_id] = {
            'param': opt_param,
            'p_name': agent.p_name, 
        }

    with open(f'fits/fit_info.pkl', 'wb')as handle:
        pickle.dump(fit_info, handle)

        

if __name__ == '__main__':

    # load data
    raw_data = pd.read_csv(f'{pth}/data/data_bandit.csv')
    data = preprocess(raw_data) 
    n_sub = 5  # fit 5 subject as an example 
    data = {k: data[k] for k in range(n_sub)}

    # fit the model to each subject
    fit_all_sub(rlddm, data)
    
    