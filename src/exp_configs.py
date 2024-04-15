""" Experimental configurations """

from utils import DATASET_CIFAR, DATASET_TINYIMAGENET, DATASET_IMAGENET


class ExpConfig():
	def __init__(self):
		self.starting_task_id = 2
		self.default_args = {
			'starting_task_id': self.starting_task_id,
			'download': True,
			'batch_mode': None,
			'use_func_regularizer': False,
			'use_ewc_weight_regularizer': False,
			'use_experience_replay': False,
			'weight_init_mode': 'continual',
			'memory_args': {
				'n_memorable_points': None,
			},
			'scale_tau_by_task_size': True,
			'starting_checkpoint_run_id': 'batch-joint',
		}

	def get_run_id(self, exp_args, seed):
		method_suffix = f'_{exp_args["n_memorable_points"]}' if 'fr' in exp_args['method'] else ''
		return f'{exp_args["method"]}{method_suffix}_{seed}'

	def get_exp_args_config(self, exp_args, seed):
		from copy import deepcopy

		args_config = deepcopy(self.default_args)
		args_config.update(deepcopy(exp_args))
		args_config['run_id'] = self.get_run_id(exp_args, seed)
		args_config['seed'] = seed

		method = args_config.pop('method')
		if 'batch' in method:
			args_config['batch_mode'] = method.split('-')[1]
		else:
			args_config['use_func_regularizer'] = 'fr' in method
			args_config['use_ewc_weight_regularizer'] = 'ewc' in method
			args_config['use_experience_replay'] = 'replay' in method
			if 'choose_m2_as_subset_of_m1' in args_config:
				args_config['choose_m2_as_subset_of_m1'] &= 'replay' in method

			if 'fr' in method:
				args_config['memory_args']['temp'] = args_config.pop('temp')
				args_config['memory_args']['n_memorable_points'] = args_config.pop('n_memorable_points')

			args_config['starting_checkpoint_run_id'] += f"_{seed}"

		return args_config


	def set_directories(self, dataset, dataset_root, result_root, imagenet_train_root=None, imagenet_val_root=None):
		self.default_args.update({
			'dataset_root': dataset_root,
			'result_root': result_root,
		})
		if dataset.lower() == DATASET_IMAGENET:
			self.default_args.update({
				'train_root': imagenet_train_root,
				'val_root': imagenet_val_root,
			})


class ExpConfigCIFAR(ExpConfig):
	def __init__(self):
		super().__init__()
		self.seeds = [0, 1, 2, 3, 4]
		self.default_args_cifar = {
			'dataset': DATASET_CIFAR,
			'n_classes': 60,
			'n_tasks': 6,
			'end_task_id': 6,
			'arch': 'CifarNet',
			'arch_args': {
				'd_in': 3,
				'd_out': 60},
			# 'epochs': 80,
			'epochs': 0,
			'optim': 'Adam',
			'optim_args': {
				'lr': 1.e-3,
				'betas': [0.9, 0.999],
				'weight_decay': 0.0},
			'batch_size': 256,
			'test_batch_size': 256,
			'val_data_frac': 0.0,
			'max_grad_norm': 100,
			'save_load_best_val_checkpoint': False,
		}
		self.default_args.update(self.default_args_cifar)


class ExpConfigTinyImageNet(ExpConfig):
	def __init__(self):
		super().__init__()
		self.seeds = [0, 1, 2]
		self.default_args_tinyimagenet = {
			'dataset': DATASET_TINYIMAGENET,
			'n_classes': 200,
			'n_tasks': 10,
			'end_task_id': 10,
			'arch': 'VGGModel',
			'arch_args': {
				'd_in': 3,
				'd_out': 200},
			# 'epochs': 70,
			'epochs': 0,
			'optim': 'SGD',
			'optim_args': {
				'lr': 1.e-2,
				'momentum': 0.9,
				'weight_decay': 0.0},
			'batch_size': 200,
			'test_batch_size': 200,
			'save_load_best_val_checkpoint': True,
			'use_lr_decay_with_early_stopping': True,
			'lr_decay_patience': 5,
			'early_stopping_patience': 10,
			'lr_decay_grace_period': 10,
		}
		self.default_args.update(self.default_args_tinyimagenet)


class ExpConfigImageNet(ExpConfig):
	def __init__(self):
		super().__init__()
		self.seeds = [0]
		self.default_args_imagenet = {
			'dataset': DATASET_IMAGENET,
			'n_classes': 1000,
			'n_tasks': 10,
			'end_task_id': 10,
			'shuffle_classes': True,
			'shuffle_classes_seed': 1993,
			'arch': 'ResNet18',
			'arch_args': {
				'num_classes': 1000},
			'epochs': 16,
			'optim': 'SGD',
			'optim_args': {
				'lr': 0.5,
				'lr_peak_epoch': 2,
				'lr_schedule_type': 'cyclic',
				'momentum': 0.9,
				'weight_decay': 5.e-5},
			'label_smoothing': 0.0,
			'batch_size': 1024,
			'test_batch_size': 512,
			'use_amp': True,
			'val_data_frac': 0.0,
			'max_grad_norm': None,
			'save_load_best_val_checkpoint': False,
		}
		self.default_args.update(self.default_args_imagenet)


class BatchConfigCIFAR(ExpConfigCIFAR):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			starting_task_id = 1,
			weight_init_mode = 'random',
		))
		self.exp_args_list = [
			dict(method = 'batch-joint'),
			dict(method = 'batch-separate'),
		]


class BatchConfigTinyImageNet(ExpConfigTinyImageNet):
	def __init__(self):
		super().__init__()
		self.seeds = [0, 1, 2, 3, 4]
		self.default_args.update(dict(
			starting_task_id = 1,
			weight_init_mode = 'random',
		))
		self.exp_args_list = [
			dict(method = 'batch-joint'),
			dict(method = 'batch-separate'),
		]


class BatchConfigImageNet(ExpConfigImageNet):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			starting_task_id = 1,
			weight_init_mode = 'random',
			epochs = 40,
		))
		self.exp_args_list = [
			dict(method = 'batch-joint'),
			dict(method = 'batch-separate'),
		]


class EWCConfigCIFAR(ExpConfigCIFAR):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			lam = 10.0,
		))
		self.exp_args_list = [dict(method = 'ewc')]


class EWCConfigTinyImageNet(ExpConfigTinyImageNet):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			lam = 330.0,
			lam2 = 85.0,
			lam3 = 45.0,
			lam4 = 30.0,
			lam5 = 20.0,
			lam6 = 15.0,
			lam7 = 15.0,
			lam8 = 10.0,
			lam9 = 10.0,
		))
		self.exp_args_list = [dict(method = 'ewc')]


class EWCConfigImageNet(ExpConfigImageNet):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			lam = 1.0,
			epochs = 40,
		))
		self.exp_args_list = [dict(method = 'ewc')]


class FRConfigCIFAR(ExpConfigCIFAR):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			temp = 2.0,
			lam = 2.0,
			choose_m2_as_subset_of_m1 = True,
			scale_tau_by_task_size = False,
		))
		self.exp_args_list = [
			dict(method = 'fr', n_memorable_points = 100, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr', n_memorable_points = 200, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr', n_memorable_points = 500, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr', n_memorable_points = 1000, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr', n_memorable_points = 2000, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr', n_memorable_points = 5000, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 100, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 200, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 500, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 1000, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 2000, tau = 0.1, tau2 = 1.0),
			dict(method = 'fr+replay', n_memorable_points = 5000, tau = 0.1, tau2 = 1.0),
			dict(method = 'ewc+fr', n_memorable_points = 100, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr', n_memorable_points = 200, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr', n_memorable_points = 500, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr', n_memorable_points = 1000, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr', n_memorable_points = 2000, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr', n_memorable_points = 5000, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 100, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 200, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 500, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 1000, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 2000, tau = 0.25, tau2 = 0.25),
			dict(method = 'ewc+fr+replay', n_memorable_points = 5000, tau = 0.25, tau2 = 0.25),
		]


class FRConfigTinyImageNet(ExpConfigTinyImageNet):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
				temp = 1.0,
				lam = 330.0,
				lam2 = 85.0,
				lam3 = 45.0,
				lam4 = 30.0,
				lam5 = 20.0,
				lam6 = 15.0,
				lam7 = 15.0,
				lam8 = 10.0,
				lam9 = 10.0,
				scale_tau_by_task_size = False,
		))
		self.exp_args_list = [
			dict(method = 'fr', n_memorable_points = 50, tau = 16.0),
			dict(method = 'fr', n_memorable_points = 100, tau = 16.0),
			dict(method = 'fr', n_memorable_points = 200, tau = 16.0),
			dict(method = 'fr', n_memorable_points = 500, tau = 2.0),
			dict(method = 'fr', n_memorable_points = 1000, tau = 2.0),
			dict(method = 'fr', n_memorable_points = 2000, tau = 16.0),
			dict(method = 'fr', n_memorable_points = 4000, tau = 12.0),
			dict(method = 'ewc+fr', n_memorable_points = 50, tau = 16.0),
			dict(method = 'ewc+fr', n_memorable_points = 100, tau = 16.0),
			dict(method = 'ewc+fr', n_memorable_points = 200, tau = 16.0),
			dict(method = 'ewc+fr', n_memorable_points = 500, tau = 32.0),
			dict(method = 'ewc+fr', n_memorable_points = 1000, tau = 16.0),
			dict(method = 'ewc+fr', n_memorable_points = 2000, tau = 16.0),
			dict(method = 'ewc+fr', n_memorable_points = 4000, tau = 12.0),

		]


class FRConfigImageNet(ExpConfigImageNet):
	def __init__(self):
		super().__init__()
		self.default_args.update(dict(
			temp = 1.0,
			choose_m2_as_subset_of_m1 = False,
			label_smoothing = 0.1,
			scale_tau_by_task_size = True
		))
		self.exp_args_list = [
			dict(method = 'fr', n_memorable_points = 200, tau = 0.16, lam = None),
			dict(method = 'fr', n_memorable_points = 500, tau = 0.16, lam = None),
			dict(method = 'fr', n_memorable_points = 1000, tau = 0.16, lam = None),
			dict(method = 'fr', n_memorable_points = 2000, tau = 0.16, lam = None),
			dict(method = 'fr', n_memorable_points = 5000, tau = 0.16, lam = None),
			dict(method = 'fr', n_memorable_points = 10000, tau = 2.0, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 200, tau = 0.12, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 500, tau = 0.16, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 1000, tau = 0.16, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 2000, tau = 0.16, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 5000, tau = 0.16, lam = None),
			dict(method = 'fr+replay', n_memorable_points = 10000, tau = 0.16, lam = None),
			dict(method = 'ewc+fr', n_memorable_points = 200, tau = 0.0, lam = 1.0),
			dict(method = 'ewc+fr', n_memorable_points = 500, tau = 0.0, lam = 1.1),
			dict(method = 'ewc+fr', n_memorable_points = 1000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr', n_memorable_points = 2000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr', n_memorable_points = 5000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr', n_memorable_points = 10000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr+replay', n_memorable_points = 200, tau = 0.0, lam = 1.0),
			dict(method = 'ewc+fr+replay', n_memorable_points = 500, tau = 0.0, lam = 1.2),
			dict(method = 'ewc+fr+replay', n_memorable_points = 1000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr+replay', n_memorable_points = 2000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr+replay', n_memorable_points = 5000, tau = 0.16, lam = 0.5),
			dict(method = 'ewc+fr+replay', n_memorable_points = 10000, tau = 0.16, lam = 0.5),
		]
