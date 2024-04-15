""" Script to create config files for experiments """

import yaml
import exp_configs

DATASETS = ['CIFAR', 'TinyImageNet', 'ImageNet']
METHODS = ['Batch', 'EWC', 'FR']

DATASET_ROOT = './src/data'
RESULT_ROOT = './src/results'
IMAGENET_TRAIN_ROOT = './src/data/imagenet_ffcv_index_400_1.0_90/train_400_1.0_90.ffcv'
IMAGENET_VAL_ROOT = './src/data/imagenet_ffcv_index_400_1.0_90/val_400_1.0_90.ffcv'


def main():
	for dataset in DATASETS:
		for method in METHODS:
			# load experiment config
			exp_config = getattr(exp_configs, f'{method}Config{dataset}')()
			exp_config.set_directories(dataset, DATASET_ROOT, RESULT_ROOT, IMAGENET_TRAIN_ROOT, IMAGENET_VAL_ROOT)

			# iterate over different exp settings and create config files
			for seed in exp_config.seeds:
				for exp_args in exp_config.exp_args_list:
					args = exp_config.get_exp_args_config(exp_args, seed)
					config_path = f'./src/configs/{dataset.lower()}/{args["run_id"]}.yaml'
					with open(config_path, "w") as f:
						yaml.dump(args, f)


if __name__=="__main__":
	main()
