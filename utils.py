import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel

from sklearn.model_selection import train_test_split, LeaveOneOut
from scipy import stats


import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset


##### standard BO
def EI_integral(mu, sigma, y_max):
    if sigma == 0:
        ei = 0
    else:
        t = (mu - y_max) / sigma
        ei = (
            (mu - y_max) * stats.norm.cdf(t) +
            sigma * stats.norm.pdf(t)
        )
    return ei

##### Benchmark Function
def generate_lhs_samples(n_samples, n_dim, l_bounds, u_bounds, seed=None):
    sampler = stats.qmc.LatinHypercube(d=n_dim, seed=seed)
    sample = sampler.random(n=n_samples)
    sample_scaled = stats.qmc.scale(sample, np.ones(n_dim)*l_bounds, np.ones(n_dim)*u_bounds)
    return sample_scaled

def ackley_function(x):
    D = x.shape[1]
    term1 = -20 * np.exp(-0.2 * np.sqrt((1/D) * np.sum(x**2, axis=1)))
    term2 = -np.exp((1/D) * np.sum(np.cos(2*np.pi*x), axis=1))
    return term1 + term2 + 20 + np.exp(1)

def rosenbrock_function(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

def levy_function(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[:, 0])**2
    term2 = np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1)
    term3 = (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)
    return term1 + term2 + term3

def styblinski_tang_function(x):
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x, axis=1)

def dixonprice_function(x):
    term1 = (x[:, 0] - 1) ** 2
    term2 = np.sum(np.arange(2, x.shape[1] + 1) * (2 * x[:, 1:]**2 - x[:, :-1])**2, axis=1)
    return term1 + term2


##### AlphaSeek

def EI(l_n1, u_n1, C, alpha, Y_max):
    if u_n1 == np.inf:
        return np.inf
    else:
        if Y_max <= l_n1 - C:
            return (l_n1 + u_n1 - 2 * Y_max) / 2
        elif l_n1 - C < Y_max <= l_n1:
            term1 = (1 - alpha) * l_n1 / 2
            term2 = alpha * ((l_n1 - Y_max) ** 2 + C ** 2 + 2 * C * Y_max) / (4 * C)
            term3 = (u_n1 - 2 * Y_max) / 2
            return term1 + term2 + term3
        elif l_n1 <= Y_max <= u_n1:
            term1 = (1 - alpha) * (u_n1 - Y_max) ** 2 / (2 * (u_n1 - l_n1))
            term2 = alpha * (2 * u_n1 + C - 2 * Y_max) / 4
            return term1 + term2
        elif u_n1 <= Y_max <= u_n1 + C:
            return alpha * (u_n1 + C - Y_max) ** 2 / (4 * C)
        else:
            return 0

def UCB(l_n1, u_n1, C, alpha, beta):
    if u_n1 == np.inf:
        return np.inf
    else:
        term1 = (u_n1 - l_n1)**2 + 3 * C * (u_n1 - l_n1) + 2 * C**2
        term2 = (u_n1 - l_n1)**2
        return (l_n1 + u_n1) / 2 + beta * np.sqrt(alpha * term1 / 6 + term2 / 12)

def sigma(l_n1, u_n1, C, alpha):
    if u_n1 == np.inf:
        return np.inf
    else:
        term1 = (u_n1 - l_n1)**2 + 3 * C * (u_n1 - l_n1) + 2 * C**2
        term2 = (u_n1 - l_n1)**2
        return np.sqrt(alpha * term1 / 6 + term2 / 12)    

def get_randomized_weighted_quantile(alpha, weights, scores):
    weights_normalized = weights / np.sum(weights)
    
    ord_s = np.argsort(scores)
    sorted_scores = scores[ord_s]
    sorted_weights = weights_normalized[ord_s]
    
    cum_weights = np.cumsum(sorted_weights)

    if alpha == 1:
        return np.inf
 
    if alpha <= cum_weights[0]:
        return sorted_scores[0]
    else:
        lq = np.where(cum_weights < alpha)[0][-1]
        uq = np.where(cum_weights > alpha)[0][0]
        Z = cum_weights[uq] - cum_weights[lq]
        p = [(cum_weights[uq]-alpha)/Z, (alpha-cum_weights[lq])/Z]
        return np.random.choice([sorted_scores[lq],sorted_scores[uq]], size=1, p=p)[0]


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.fc4 = nn.Linear(16, 1)
        self.initialize_weights() 

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
      
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.layer_norm(x)
        x = self.fc4(x)  
        return x


def weighted_conformal_quantile(
        df_X, y,                 
        df_pool_X,                     
        alpha=0.1,               
        model_type='rf',         
        model_path=None,         
        nn_epochs=100,           
        lr=1e-4,                    
    ):

    clf = RandomForestClassifier(n_estimators=100,class_weight='balanced', criterion='entropy', min_weight_fraction_leaf=0.1).fit(pd.concat([df_X, df_pool_X]),  [0]*len(df_X)+[1]*len(df_pool_X))
    prob_train_0 = clf.predict_proba(df_X)[:, 0]
    prob_train_1 = clf.predict_proba(df_X)[:, 1]
    prob_pool_0 = clf.predict_proba(df_pool_X)[:, 0]
    prob_pool_1 = clf.predict_proba(df_pool_X)[:, 1]
    w_train = np.divide(prob_train_1, prob_train_0, out=np.ones_like(prob_train_1) * np.inf, where=prob_train_0 != 0)
    w_pool = np.divide(prob_pool_1, prob_pool_0, out=np.ones_like(prob_pool_1) * np.inf, where=prob_pool_0 != 0)
    w_train[w_train >= 10] = 10
    w_pool[w_pool >= 10] = 10

    X_t, X_cal, y_t, y_cal = train_test_split(
        df_X, y,
        test_size=0.5
    )
    cal_index = np.array(y_cal.index)


    if model_type.lower() == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            criterion='absolute_error'
        )
        model.fit(X_t, y_t)

        y_cal_pred = model.predict(X_cal)
        y_pool_pred = model.predict(df_pool_X)

    elif model_type.lower() == 'gp':
        model = GaussianProcessRegressor(
            kernel=ConstantKernel() * RBF() + WhiteKernel(), alpha=0, normalize_y=True
        )
        model.fit(X_t, y_t)

        y_cal_pred = model.predict(X_cal)
        y_pool_pred = model.predict(df_pool_X)

    elif model_type.lower() == 'nn':
        model = NN() 
        if model_path is not None and os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))


        X_train_tensor = torch.tensor(X_t.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_t.values, dtype=torch.float32).view(-1, 1)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=len(X_train_tensor), 
            shuffle=False
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

        model.train()
        for _ in range(nn_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_cal_pred = model(torch.tensor(X_cal.values, dtype=torch.float32)).numpy().ravel()
            y_pool_pred = model(torch.tensor(df_pool_X.values, dtype=torch.float32)).numpy().ravel()

    score = np.abs(y_cal_pred - y_cal.values.ravel())
    sorted_indices = np.argsort(score)
    sorted_index = cal_index[sorted_indices]    
    sorted_score = score[sorted_indices]

    w_cal = w_train[sorted_index]

    y_q = [get_randomized_weighted_quantile(1-alpha, np.hstack([w_cal, w_]), np.hstack([sorted_score, np.inf])) for w_ in w_pool]

    return y_pool_pred, y_q
