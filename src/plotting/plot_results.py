import plotting_utils
from utils import DATASET_CIFAR, DATASET_TINYIMAGENET, DATASET_IMAGENET

# Split-CIFAR
plotting_utils.ResultHandler(DATASET_CIFAR).plot_limited_memory_results('cifar_acc.pdf')
plotting_utils.ResultHandler(DATASET_CIFAR, first_task=0).plot_results_across_tasks_multiple('cifar_forgetting.pdf')

# Split-TinyImageNet
plotting_utils.ResultHandler(DATASET_TINYIMAGENET).plot_limited_memory_results('tinyimagenet_acc.pdf')
plotting_utils.plot_tinyimagenet_baseline_results('tinyimagenet_baselines.pdf')

# ImageNet-1000
plotting_utils.ResultHandler(DATASET_IMAGENET).plot_limited_memory_results('imagenet_acc.pdf')
plotting_utils.ResultHandler(DATASET_IMAGENET, first_task=0).plot_results_across_tasks_multiple('imagenet_forgetting.pdf')
