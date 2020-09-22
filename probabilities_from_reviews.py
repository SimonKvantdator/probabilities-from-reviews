import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom

import os
import re # regular expressions
import datetime

# load configurations from json
import json
with open('config.json', 'r') as config_file:
	configs = json.loads(config_file.read())
	config_file.close()

# for plotting nice plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner # for corner plots
fsize = 16
font = {'size': fsize}
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=fsize)
mpl.rc('ytick', labelsize=fsize)
mpl.rc('text', usetex=configs['use_tex'])

# for MCMC
from emcee import EnsembleSampler as ensembleSampler
from multiprocessing import Pool


### load reviews from json file ###
with open('reviews.json', 'r') as reviews_file:
	reviews = json.loads(reviews_file.read())
	reviews_file.close()

review_frequencies = []
for i in reviews:
	review_frequencies += [reviews[i]]
nbr_reviews = sum(review_frequencies)

# one parameter for the probability of each review score, but since
# the sum of the probabilities = 1, the last parameter is determined
nbr_parameters = len(review_frequencies) - 1


### define probabilities ###
def log_likelihood(p):
	"""
	Returns the log of the probability of the reviews being observed given parameters p.
	"""
	p_full = np.append(p, [1.0 - sum(p)]) # one parameter for the probability of each review score
	probability_list = binom.pmf(review_frequencies, nbr_reviews, p_full)
	log_probability_sum = np.sum(np.log(probability_list))
	
	if np.isnan(log_probability_sum):
		return -np.inf
	else:
		return log_probability_sum

def log_flat_prior(p):
	"""
	Returns log of a flat pdf
	"""
	p_full = np.append(p, [1.0 - sum(p)]) # one parameter for the probability of each review score

	if ((0.0 <= p_full) * (p_full <= 1.0)).all():
		return 0.0
	else:
		return -np.inf

def log_posterior(p):
	"""
	Returns log of posterior pdf using a flat prior pdf
	"""
	return log_flat_prior(p) + log_likelihood(p) # choose your prior here


### define MCMC sampler ###
def my_sampling(dim, log_posterior, nbr_walkers=configs['nbr_walkers'], nbr_warmup=configs['nbr_warmup'], nbr_samples=configs['nbr_samples']):
	"""
	Returns sample chain from MCMC given dimension dim of problem and a logarithmic pdf
	log_posterior.
	"""
	initial_positions = np.random.rand(nbr_walkers, dim)
	initial_positions /= np.expand_dims(np.sum(initial_positions, axis=1), axis=1) + np.random.rand(nbr_walkers, 1) # make sure sum(p_full) = 1
	sampler = ensembleSampler(nbr_walkers, dim, log_posterior, pool=Pool())

	pos, tr, pr = sampler.run_mcmc(initial_positions, nbr_warmup)
	sampler.reset()
	sampler.run_mcmc(pos, nbr_samples);
		
	return sampler.flatchain
    
### run MCMC sampler ###
samples = my_sampling(nbr_parameters, log_posterior)
samples_full = np.append(samples, np.expand_dims(1.0 - samples.sum(axis=1), axis=1), axis=1) # add last parameter to samples, even though it is determined by the other ones

# make histogram corner plot of MCMC result
scores = np.array([int(score) for score in reviews])
if configs['show_figs'] or configs['save_figs']:
	tex_labels = [r'$p_{' + str(i + 1) + r'}$' for i in range(nbr_parameters + 1)]
	fig1 = corner.corner(samples_full,
		quantiles=[0.16, 0.5, 0.84],
		bins = configs['nbr_bins'],
		show_titles=True,
		title_fmt='.3f',
		labels=tex_labels)
	plt.tight_layout()

	# make plot of probability of rating
	ratings = samples_full @ scores.T
	fig2, ax = plt.subplots(1, 1, figsize=(6, 4))
	ax.hist(ratings,
		density=True,
		bins=configs['nbr_bins'])
	ax.set_xlabel(r'rating')
	ax.set_ylabel(r'$p$(rating)')
	ax.set_xlim([1.0, 5.0])
	plt.tight_layout()


### estimate p ###
# mean estimate of p
p_mean = np.mean(samples, axis=0)
p_full_mean = np.mean(samples_full, axis=0)

# max likelihood estimate of p
def negative_log_likelihood(p):
	return -log_likelihood(p)
optimizer_result = minimize(negative_log_likelihood, p_mean, method='Nelder-Mead')
p_full_max_likelihood = np.append(optimizer_result['x'], [1.0 - sum(optimizer_result['x'])])

results = {} # initialize dict
results['p_mean'] = p_full_mean.tolist()
results['rating_mean'] = p_full_mean @ scores.T
results['p_max_likelihood'] = p_full_max_likelihood.tolist()
results['rating_max_likelihood'] = p_full_max_likelihood @ scores.T


### save results ###
# remove comments from config file
commentless_configs = {} # initialize
for tmp in configs:
	if re.match('__', tmp) is None:
		commentless_configs[str(tmp)] = configs[str(tmp)]

# create folder named with the current time to save figures and data in there
folder_name = str(datetime.datetime.now())
os.mkdir(folder_name)

# save configurations and results
configs_and_results = {} # initialize
configs_and_results['configs'] = commentless_configs
configs_and_results['results'] = results
with open(f'{folder_name}/configs_and_results.json', 'w') as configs_and_results_file:
	json.dump(configs_and_results, configs_and_results_file, indent=4)
	configs_and_results_file.close()

if configs['save_figs']:
	# save figures
	fig1.savefig(f'{folder_name}/mcmc_histogram_corner_plot.pdf')
	fig2.savefig(f'{folder_name}/ratings_histogram.pdf')

# show all figures
if configs['show_figs']:
	plt.show()
