""" Utility methods for plotting """

import numpy as np
import pickle
from collections import defaultdict
from pathlib import Path

from exp_configs import FRConfigCIFAR, FRConfigTinyImageNet, FRConfigImageNet
from utils import DATASET_CIFAR, DATASET_TINYIMAGENET, DATASET_IMAGENET

RES_ROOT_LOCAL = './src/results'
PLOT_ROOT = './src/plotting/plots'

class ResultHandler:
	def __init__(self, dataset, verbose=False, first_task=1):
		if dataset == DATASET_CIFAR:
			self.exp_config = FRConfigCIFAR()
		elif dataset == DATASET_TINYIMAGENET:
			self.exp_config = FRConfigTinyImageNet()
		elif dataset == DATASET_IMAGENET:
			self.exp_config = FRConfigImageNet()

		self.dataset = dataset
		self.seeds = self.exp_config.seeds
		end_task_id = self.exp_config.default_args['end_task_id']
		self.task_ids = range(1, end_task_id + 1)

		self.plot_root = Path(PLOT_ROOT) / self.dataset
		self.verbose = verbose
		self.first_task = first_task

		self.results = get_results_dict()
		self.data = {t: [] for t in self.task_ids[self.first_task:]}
		self.exp_args_all = {t: [] for t in self.task_ids[self.first_task:]}

		self.load_results()


	def load_all_task_results(self, run_id):
		res = {}
		for task in self.task_ids:
			exp_fname = Path(RES_ROOT_LOCAL) / self.dataset.lower() / 'exp' / run_id / f'post_task{task}' / 'results.pkl'
			if exp_fname.exists():
				res[task] = load_results_from_file(exp_fname)
		return res


	def load_results(self):
		exp_args_list = self.exp_config.exp_args_list
		exp_args_list += [dict(method='batch-joint'), dict(method='batch-separate'), dict(method='ewc')]
		results = {}
		for exp_args in exp_args_list:
			for seed in self.seeds:
				run_id = self.exp_config.get_run_id(exp_args, seed)
				results[seed] = self.load_all_task_results(run_id)

			for t in self.task_ids[self.first_task:]:
				if exp_args['method'] == 'batch-separate':
					data = [np.mean([results[s][t_][f'test_after_task{t_}'][f'task{t_}']['acc'] for t_ in range(self.task_ids[0], t+1)]) for s in self.seeds]
				else:
					data = [results[s][t][f'test_after_task{t}']['avg']['acc'] if t in results[s] else np.nan for s in self.seeds]

				self.data[t].append(data)
				self.exp_args_all[t].append(exp_args)

		return self


	def get_baseline_mean(self, task_id, method):
		import numpy as np
		for data_, exp_args in zip(self.data[task_id], self.exp_args_all[task_id]):
			if exp_args['method'] == method:
				return np.mean(data_)


	def get_baseline_data(self, task_id, method):
		for data_, exp_args in zip(self.data[task_id], self.exp_args_all[task_id]):
			if exp_args['method'] == method:
				return data_


	def plot_limited_memory_results(self, plot_fname):
		import matplotlib.pyplot as plt
		from tueplots import bundles
		from matplotlib.ticker import MultipleLocator
		import matplotlib.ticker as mticker

		tasks = [self.task_ids[-1]]
		n_plots = len(tasks)

		rel_width = 0.49
		plt.rcParams.update(bundles.neurips2022(rel_width=rel_width, nrows=n_plots))

		params = {'axes.labelsize': 'small',
				'xtick.labelsize': 'x-small',
				'ytick.labelsize': 'x-small'}
		plt.rcParams.update(params)

		if self.dataset == DATASET_CIFAR:
			n_memorable_points = [100, 200, 500, 1000, 2000, 5000]
		elif self.dataset == DATASET_TINYIMAGENET:
			n_memorable_points = [50, 100, 200, 500, 1000, 2000, 4000]
		elif self.dataset == DATASET_IMAGENET:
			n_memorable_points = [200, 500, 1000, 2000, 5000, 10000]
		xmin, xmax = n_memorable_points[0], n_memorable_points[-1]

		_, axs = plt.subplots(n_plots, 1)
		if n_plots == 1:
			axs = [axs]

		y_fac = 100.0	# plot percentages
		for t, ax in zip(tasks, axs):
			ax.hlines(y=y_fac * self.get_baseline_mean(t, 'batch-joint'), xmin=xmin, xmax=xmax, color='#ff7f00', linestyle='--', lw=1)
			ax.hlines(y=y_fac * self.get_baseline_mean(t, 'batch-separate'), xmin=xmin, xmax=xmax, color='#4daf4a', linestyle='dotted', lw=1)
			ax.hlines(y=y_fac * self.get_baseline_mean(t, 'ewc'), xmin=xmin, xmax=xmax, color='#984ea3', linestyle='-.', lw=1)
			plot_lines(ax, t, self.data, self.exp_args_all, n_memorable_points, len(self.seeds), self.dataset, y_fac)

			ax.set_xscale('log')
			x_percentages = {
				DATASET_CIFAR: [0.01, 0.03, 0.10, 0.333],
				DATASET_TINYIMAGENET: [0.01, 0.03, 0.10, 0.45],
				DATASET_IMAGENET: [0.0015, 0.005, 0.02, 0.075],
			}
			total_train_data = {
				DATASET_CIFAR: 75000,
				DATASET_TINYIMAGENET: 80000,
				DATASET_IMAGENET: 1200000,
			}
			x_perc_data = [x_perc * total_train_data[self.dataset] / (len(self.task_ids) - 1) for x_perc in x_percentages[self.dataset]]
			ax.set_xticks(x_perc_data)
			if self.dataset == DATASET_IMAGENET:
				ax.set_xticklabels([f'{100 * x_perc:.1f}\%' for x_perc in x_percentages[self.dataset]])
			else:
				ax.set_xticklabels([f'{100 * x_perc:.0f}\%' for x_perc in x_percentages[self.dataset]])
			base = 20 if self.dataset == DATASET_IMAGENET else 5
			ax.yaxis.set_major_locator(MultipleLocator(base=base))
			ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
			ax.set_xlabel('Memory size (\% of data)')
			ax.set_ylabel('Test acc. (avg. over tasks)')

			ax.xaxis.label.set_color('gray')
			ax.yaxis.label.set_color('gray')
			ax.tick_params(axis='x', colors='gray')
			ax.tick_params(axis='y', colors='gray')
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			for spine in ax.spines.values():
				spine.set_edgecolor('grey')

		plt.savefig(self.plot_root / plot_fname, bbox_inches='tight', pad_inches=0.0)


	def plot_results_across_tasks_multiple(self, plot_fname):
		import matplotlib.pyplot as plt
		from tueplots import bundles

		rel_width = 0.49
		plt.rcParams.update(bundles.neurips2022(rel_width=rel_width, nrows=1, ncols=1))

		params = {'axes.labelsize': 'small',
				'xtick.labelsize': 'x-small',
				'ytick.labelsize': 'x-small'}
		plt.rcParams.update(params)

		if self.dataset == DATASET_CIFAR:
			mem = [100, 5000]
		else:
			mem = [10000]

		_, axes = plt.subplots(nrows=1, ncols=len(mem), sharex=True, sharey=True)
		if self.dataset == DATASET_CIFAR:
			axes = axes.flatten()
		else:
			axes = [axes]
		for m, ax in zip(mem, axes):
			self.plot_results_across_tasks(m, ax, self.dataset)

			ax.xaxis.label.set_color('gray')
			ax.yaxis.label.set_color('gray')
			ax.tick_params(axis='x', colors='gray')
			ax.tick_params(axis='y', colors='gray')
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			for spine in ax.spines.values():
				spine.set_edgecolor('grey')

		plt.savefig(self.plot_root / plot_fname, bbox_inches='tight', pad_inches=0.0)


	def plot_results_across_tasks(self, mem, ax, dataset):
		import numpy as np
		import matplotlib.ticker as mticker

		methods = [
			'fr+replay',
			'ewc+fr',
			'ewc+fr+replay',
			'fr',
		]
		baselines = [
			'batch-joint',
			'batch-separate',
			'ewc',
		]
		all_methods = methods + baselines

		methods_to_plot = ['batch-joint', 'ewc', 'fr', 'ewc+fr+replay']

		colors = {
			'ewc+fr+replay': '#e41a1c',
			'ewc+fr': '#377eb8',
			'fr+replay': '#377eb8',
			'fr': '#377eb8',
			'batch-joint': '#ff7f00',
			'batch-separate': '#4daf4a',
			'ewc': '#984ea3',
		}

		markers = {
			'ewc+fr+replay': 's',
			'ewc+fr': 'o',
			'fr+replay': 'o',
			'fr': 's',
			'batch-joint': 'x',
			'batch-separate': '^',
			'ewc': 'o',
		}

		ms = {
			'ewc+fr+replay': 5,
			'ewc+fr': 3,
			'fr+replay': 3,
			'fr': 5 if dataset == DATASET_IMAGENET else 3,
			'batch-joint': 3,
			'batch-separate': 3,
			'ewc': 3,
		}

		data_by_task = {method: [self.get_baseline_data(1, 'batch-separate')] for method in all_methods}
		for t in self.task_ids[1:]:
			for baseline in baselines:
				data_by_task[baseline].append(self.get_baseline_data(t, baseline))

			for method in methods:
				for data_, exp_args_ in zip(self.data[t], self.exp_args_all[t]):
					if exp_args_['method'] == method and exp_args_['n_memorable_points'] == mem:
						data_by_task[method].append(data_)

		# fr as lower bound of ewc+fr (lam=0.0)
		if np.mean(data_by_task['ewc+fr+replay']) < np.mean(data_by_task['fr+replay']):
			data_by_task['ewc+fr+replay'] = data_by_task['fr+replay']

		# relative to batch joint
		for method in all_methods:
			if method == 'batch-joint':
				continue
			data_by_task[method] = np.array(data_by_task[method]) - np.array(data_by_task['batch-joint'])
		data_by_task['batch-joint'] = np.array(data_by_task['batch-joint']) - np.array(data_by_task['batch-joint'])

		# compute forgetting
		forgetting = {method: [[0] * len(self.seeds)] for method in all_methods}
		for method in all_methods:
			for t in self.task_ids[1:]:
				diffs = []
				for t_ in range(1, t):
					diffs.append(np.array(data_by_task[method][t-1]) - np.array(data_by_task[method][t_-1]))
				forgetting[method].append(np.mean(diffs, axis=0))
			forgetting[method] = np.array(forgetting[method])

		# relative to batch joint
		for method in all_methods:
			if method == 'batch-joint':
				continue
			forgetting[method] -= forgetting['batch-joint']
		forgetting['batch-joint'] -= forgetting['batch-joint']

		y_fac = 100.0	# plot percentages
		for method in all_methods:
			if method not in methods_to_plot:
				continue
			mean = y_fac * np.mean(forgetting[method], axis=1)
			std_err = y_fac * np.std(forgetting[method], axis=1) / np.sqrt(len(self.seeds))
			if method == 'batch-joint':
				ax.plot(self.task_ids, mean, ls='--', lw=1, color=colors[method])
			else:
				ax.plot(self.task_ids, mean, marker=markers[method], color=colors[method], mfc='white', ms=ms[method], lw=1)
			ax.fill_between(self.task_ids, mean-std_err, mean+std_err, alpha=0.3, color=colors[method])

		ax.set_xticks(self.task_ids)
		ax.set_xticklabels(self.task_ids)
		ax.set_xlabel('Number of tasks')
		ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
		ax.set_ylabel('Forgetting (avg. over tasks)')


def plot_lines(ax, t, data, exp_args, n_memorable_points, n_seeds, dataset, y_fac):
	import numpy as np

	if dataset == DATASET_TINYIMAGENET:
		methods = [
			'ewc+fr+replay',
			'ewc+fr',
			'fr+replay',
			'fr',
		]
		methods_to_plot = ['fr', 'ewc+fr']
	else:
		methods = {
			'ewc+fr+replay',
			'fr',
			'fr+replay',
			'ewc+fr',
		}
		methods_to_plot = ['ewc+fr+replay', 'ewc+fr', 'fr']

	markers = {
		'ewc+fr+replay': 's',
		'ewc+fr': 's' if dataset == DATASET_TINYIMAGENET else 'o',
		'fr+replay': 's',
		'fr': 'o' if dataset == DATASET_TINYIMAGENET else 's',
	}

	ms = {
		'ewc+fr+replay': 5,
		'ewc+fr': 5 if dataset == DATASET_TINYIMAGENET else 3,
		'fr+replay': 5,
		'fr': 3 if dataset == DATASET_TINYIMAGENET else 5,
	}

	colors = {
		'ewc+fr+replay': '#e41a1c',
		'ewc+fr': '#e41a1c' if dataset == DATASET_TINYIMAGENET else '#377eb8',
		'fr+replay': '#377eb8',
		'fr': '#377eb8',
	}

	data_by_memory = {method: [] for method in methods}
	for method in methods:
		for m in n_memorable_points:
			found = False
			for data_, exp_args_ in zip(data[t], exp_args[t]):
				if exp_args_['method'] == method and exp_args_['n_memorable_points'] == m:
					data_by_memory[method].append(data_)
					found = True
			if not found:
				data_by_memory[method].append([np.nan] * n_seeds)

	# fr as lower bound of ewc+fr (lam=0.0)
	for i in range(len(n_memorable_points)):
		for use_experience_replay in [False, True]:
			suffix = '+replay' if use_experience_replay else ''
			if np.mean(data_by_memory[f'ewc+fr{suffix}'][i]) < np.mean(data_by_memory[f'fr{suffix}'][i]):
				data_by_memory[f'ewc+fr{suffix}'][i] = data_by_memory[f'fr{suffix}'][i]

	# fr as lower bound of fr+replay
	for i in range(len(n_memorable_points)):
		if np.isnan(np.mean(data_by_memory['ewc+fr+replay'][i])):
			data_by_memory['ewc+fr+replay'][i] = data_by_memory['ewc+fr'][i]

	for method in methods:
		if method not in methods_to_plot:
			continue
		mean = y_fac * np.mean(data_by_memory[method], axis=1)
		std_err = y_fac * np.std(data_by_memory[method], axis=1) / np.sqrt(n_seeds)
		ax.plot(n_memorable_points, mean, marker=markers[method], color=colors[method], mfc='white', ms=ms[method], lw=1)
		ax.fill_between(n_memorable_points, mean-std_err, mean+std_err, alpha=0.3, color=colors[method])


def plot_tinyimagenet_baseline_results(plot_fname):
	""" Split-TinyImageNet baseline numbers taken from Tables 4 and 5 in
		A continual learning survey: Defying forgetting in classification tasks (De Lange et al.)
		https://arxiv.org/abs/1909.08383
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	from tueplots import bundles
	from pathlib import Path
	import matplotlib.ticker as mticker

	rel_width = 0.49
	plt.rcParams.update(bundles.neurips2022(rel_width=rel_width, nrows=1))

	params = {'axes.labelsize': 'small',
		'xtick.labelsize': 'x-small',
		'ytick.labelsize': 'x-small'}
	plt.rcParams.update(params)

	n_memorable_points = [500, 1000]

	_, ax = plt.subplots(1, 1)

	for mem in n_memorable_points:
		ax.axvline(x=mem, color='grey', linewidth=0.7, alpha=0.5)

	baselines = {
		'PackNet': (49.1, 'tab:blue'),
		'iCaRL': ([47.3, 48.8], 'tab:pink'),
		'MAS': (46.9, 'tab:green'),
		'EBLL': (45.3, 'tab:purple'),
		'HAT': (43.6, 'tab:brown'),
		'R-FM': ([37.3, 42.4], 'tab:cyan'),
		'EWC': (42.4, 'tab:gray'),
		'LwF': (41.9, 'tab:olive'),
		'GEM': ([45.1, 41.8], 'tab:orange'),
		'R-PM': ([36.1, 38.7], 'tab:green'),
		'mode-IMM': (36.9, 'tab:orange'),
		'SI': (33.9, 'tab:pink'),
	}

	ms_baseline = 3
	ms_ours = 6
	for baseline, (score, col) in baselines.items():
		if isinstance(score, list):
			ls = '-'
		else:
			score = [score, score]
			ls = 'dotted'
		ax.plot(n_memorable_points, np.array(score), color=col, mfc='white', marker='o', ms=ms_baseline, lw=1, ls=ls, label=baseline)

	fr_mean = np.array([49.096667, 51.24])
	fr_std_err = np.array([0.167221, 0.298031])
	ax.plot(n_memorable_points, fr_mean, color='tab:red', marker='*', ms=ms_ours, mfc='white', lw=1, label='\\textbf{Ours}')
	ax.fill_between(n_memorable_points, fr_mean-fr_std_err, fr_mean+fr_std_err, color='tab:red', alpha=0.3)

	ax.set_xlim([425, 1075])
	ax.set_xticks(n_memorable_points)
	mem_perc_data = [m * 9 / 80000 * 100 for m in n_memorable_points]
	ax.set_xticklabels([f'{m:.1f}\%' for m in mem_perc_data])
	ax.set_xlabel('Memory size (\% of data)')
	ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

	ax.xaxis.label.set_color('gray')
	ax.yaxis.label.set_color('gray')
	ax.tick_params(axis='x', colors='gray')
	ax.tick_params(axis='y', colors='gray')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	for spine in ax.spines.values():
		spine.set_edgecolor('grey')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

	plot_root = Path(PLOT_ROOT) / DATASET_TINYIMAGENET
	plt.savefig(plot_root / plot_fname, bbox_inches='tight', pad_inches=0.0)


def load_results_from_file(fname):
	with open(fname, 'rb') as res_file:
		res = pickle.load(res_file)
	return res


def get_results_dict():
	""" Create infinitely-nested default dictionary """
	def ndd():
		return defaultdict(ndd)
	return ndd()
