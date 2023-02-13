import numpy  as np
import pandas as pd
import csv
import math

import pymc  as pm

RANDOM_SEED = 58
rng = np.random.default_rng(RANDOM_SEED)

############################################################
## Fuctions. I'll do these as class objects at some point ##
############################################################
def run_regression(xs, ys):
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        epsilon = pm.LogNormal("epsilon", 0, 10)
        y0 = pm.Normal("y0", np.mean(ys), sigma=np.std(ys))
        beta = pm.Exponential("beta", (np.max(xs) - np.min(xs))/(np.max(ys) - np.min(ys)))

        prediction = y0 - beta * xs
        
        # Define likelihood
        likelihood = pm.Normal("mod", mu=prediction, sigma=epsilon, observed=ys)
    
        # Inference!
        # draw 1000 posterior samples using NUTS sampling
        trace = pm.sample(1000, return_inferencedata=True)

    return(trace)


def conpare_gradients(trace1, trace2):
    beta1 = trace1.posterior["beta"].values 
    beta2 = trace2.posterior["beta"].values

    min_beta = np.min(np.append(beta1, beta2))
    max_beta = np.max(np.append(beta1, beta2))
    nbins = int(np.ceil(np.sqrt(beta1.size))) 

    bins = np.linspace(min_beta, max_beta, nbins)

    dist1 = np.histogram(beta1, bins)[0]
    dist2 = np.histogram(beta2, bins)[0]

    return np.sqrt(np.sum(dist1*dist2))/np.sum(dist1)


##############################################
## The actual example you'll need to apdate ##
##############################################

## Import example data
dat = pd.read_csv('data/TOA_vs_TAS.csv')

x1 = dat['TOA_with'].values
y1 = dat['Temp_with'].values

x2 = dat['TOA_without'].values
y2 = dat['Temp_without'].values

## Run Regressiob
trace1 = run_regression(x1, y1)
trace2 = run_regression(x2, y2)

p_ish_value = conpare_gradients(trace1, trace2)

## Compare data
print("\n   Prob. two gradients are the same:", np.round(p_ish_value, 3), "\n")




