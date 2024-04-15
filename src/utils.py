import os
import random
import pickle
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import datasets, transforms
import numpy as np

import models


DATASET_CIFAR10 = 'CIFAR-10'
DATASET_CIFAR100 = 'CIFAR-100'
# Split-CIFAR
DATASET_CIFAR = 'CIFAR'
# Split-TinyImageNet
DATASET_TINYIMAGENET = 'TinyImageNet'
# ImageNet-1000
DATASET_IMAGENET = 'ImageNet'

DATASETS = {
    # [dataset_class, n_classes, mean, std]
    DATASET_CIFAR10:
    [datasets.CIFAR10, 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
    DATASET_CIFAR100:
    [datasets.CIFAR100, 100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
    DATASET_CIFAR:
    [None, 60, None, None],
    DATASET_TINYIMAGENET:
    [None, 200, None, None],
    DATASET_IMAGENET:
    [None, 1000, None, None],
}

_default_entries = [
    'epoch',
    'iteration',
    'train_loss',
    'test_loss',
    'train_acc@1',
    'train_acc@5',
    'test_acc@1',
    'test_acc@5',
    'learning_rate'
]

STATUS_COMPLETE = 'Complete'
STATUS_INCOMPLETE = 'Incomplete'
STATUS_INFEASIBLE = 'Infeasible'


class Logger:
    def __init__(self, entries=None):
        self.update_templates(entries)
        self.status_incomplete()

    def update_templates(self, entries=None):
        # Prepare print report format used in Chainer
        # https://github.com/chainer/chainer/blob/v7.1.0/chainer/training/extensions/print_report.py
        entries = _default_entries if entries is None else entries
        entry_widths = [max(10, len(s)) for s in entries]
        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))

        self._templates = templates
        self._header = '  '.join(('{:%d}' % w
                                  for w in entry_widths)).format(*entries)

    def _status_update(self, status):
        self._status = status

    def status_complete(self):
        self._status_update(STATUS_COMPLETE)

    def status_incomplete(self):
        self._status_update(STATUS_INCOMPLETE)

    def status_infeasible(self):
        self._status_update(STATUS_INFEASIBLE)

    def print_header(self):
        print(self._header)

    def print_report(self, log):
        # print report
        report = ''
        for entry, template, empty in self._templates:
            if entry in log:
                try:
                    report += template.format(log[entry])
                except (TypeError, ValueError):
                    assert len(log[entry]) <= len(empty)
                    report += log[entry] + empty[len(log[entry]):]
            else:
                report += empty
        print(report)


def get_dataset(dataset,
                dataset_root='./data',
                download=True,
                random_crop=False,
                random_crop_args=None,
                random_horizontal_flip=False):
    # Collect dataset information
    dataset_class, _, mean, std = DATASETS[dataset]

    # Setup data augmentation & data pre processing
    train_transforms, test_transforms = [], []
    if random_crop:
        train_transforms.append(
            transforms.RandomCrop(random_crop_args['size'],
                                  padding=random_crop_args.get('padding', None)))
    if random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    normalize = transforms.Normalize(mean, std)
    train_transforms.append(normalize)
    train_transform = transforms.Compose(train_transforms)

    test_transforms = [transforms.ToTensor(), normalize]
    test_transform = transforms.Compose(test_transforms)

    # Setup kwargs to dataset_class
    kwargs = dict(root=dataset_root, download=download)
    train_kwargs = dict(transform=train_transform, **kwargs)
    test_kwargs = dict(transform=test_transform, **kwargs)

    train_dataset = dataset_class(train=True, **train_kwargs)
    test_dataset = dataset_class(train=False, **test_kwargs)

    return train_dataset, test_dataset


def get_data_loader(dataset=None,
                    train_dataset=None,
                    test_dataset=None,
                    batch_size=32,
                    test_batch_size=None,
                    n_workers=4,
                    world_size=1,
                    **dataset_kwargs):

    if train_dataset is None or test_dataset is None:
        assert dataset is not None
        train_dataset, test_dataset = get_dataset(dataset, **dataset_kwargs)

    is_distributed = world_size > 1
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=n_workers,
                              pin_memory=True,
                              drop_last=False)

    if test_batch_size is None:
        test_batch_size = batch_size
    test_sampler = DistributedSampler(test_dataset) if is_distributed else None
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             sampler=test_sampler,
                             num_workers=n_workers,
                             pin_memory=True)

    return train_loader, test_loader


def get_model(arch_name, arch_kwargs, dataset):
    arch_class = getattr(models, arch_name, None)
    model = arch_class(**arch_kwargs)
    arch_kwargs['dataset'] = dataset
    arch_kwargs['n_classes'] = DATASETS[dataset][1]
    return model, arch_kwargs


def get_optimizer(optim_name, params, optim_kwargs):
    optim_kwargs['params'] = params
    optim_class = getattr(torch.optim, optim_name, None)
    optimizer = optim_class(**optim_kwargs)
    optim_kwargs.pop('params')
    return optimizer, optim_kwargs


def get_device_and_comm_rank(no_cuda=False, distributed_backend='nccl', local_rank=-1, run_id=None, master_port=None, master_port_default=6000):
    if local_rank == -1:
        # [COMM] get MPI rank
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", '1'))
    else:
        rank = local_rank
        world_size = torch.cuda.device_count()
    is_distributed = world_size > 1

    if is_distributed:
        success = False
        n_retries = 0
        while not success:
            if local_rank == -1:
                # [COMM] initialize process group
                master_ip = os.getenv('MASTER_ADDR', 'localhost')
                if master_port is None and os.getenv('MASTER_PORT') is not None:
                    master_port = os.getenv('MASTER_PORT')
                else:
                    master_port = str(master_port_default + master_port)
                init_method = 'tcp://' + master_ip + ':' + master_port
            else:
                init_method = 'env://'
            try:
                print(f'Initializing DDP process group @ {init_method}...')
                dist.init_process_group(backend=distributed_backend,
                                        world_size=world_size,
                                        rank=rank,
                                        init_method=init_method)
                success = True
            except Exception as E:
                print(E)
                os.environ['MASTER_PORT'] = str((sum([ord(c) for c in run_id]) + n_retries) % 65536)
                n_retries += 1

    if not no_cuda and torch.cuda.is_available():
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, rank, world_size


def cross_entropy(predictions, targets, class_ids: List[int] = None, weight: Optional[torch.Tensor] = None, reduction: str = "mean", label_smoothing: float = 0.0):
    # Define cross-entropy function depending on if we have hard or soft labels
    if len(targets.shape) == 1:
        ce_function = F.cross_entropy
        ce_kwargs = dict()
    else:
        ce_function = cross_entropy_with_probs
        ce_kwargs = dict(reduction=reduction)
    if label_smoothing > 0.0:
        ce_kwargs.update(dict(label_smoothing=label_smoothing))

    if class_ids is not None:
        predictions = predictions.gather(1, class_ids)
    return ce_function(predictions, targets, weight=weight, **ce_kwargs)


def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", "task-equalized_mean" indicating whether to return one loss
        per data point, the mean loss, the sum of losses, or the task-equalized mean of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted

    Source: https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
    """

    assert input.shape == target.shape, "Inputs and targets must have same shape!"

    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        ce_kwargs = dict(label_smoothing=label_smoothing) if label_smoothing > 0.0 else dict()
        y_loss = F.cross_entropy(input, target_temp, reduction="none", **ce_kwargs)
        if weight is not None:
            if isinstance(weight, torch.Tensor) and len(weight.shape) == 2:
                y_loss = y_loss * weight[:, y]
            else:
                y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def top1_correct(inputs: Tensor, targets: Tensor, class_ids: List[int] = None):
    return topk_correct(inputs, targets, (1,), class_ids)[0]


def topk_correct(inputs: Tensor, targets: Tensor, topk=(1,), class_ids: List[int] = None):
    if class_ids is not None:
        inputs = inputs.gather(1, class_ids)

    with torch.no_grad():
        maxk = max(topk)

        _, pred = inputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def weight_reset(module):
	if hasattr(module, 'reset_parameters'):
		module.reset_parameters()


def set_seed(seed):
	""" set random seed """

	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True  # NOTE: this will slow down training


def set_lr(optimizer, count, decay_patience=5, early_stopping_patience=10, time=0.0, decay_factor=0.1, is_master=False):
	"""
	Early stop or decay learning rate by the given decay factor based on count.
	:param lr:              Current learning rate
	:param count:           Amount of times of not increasing accuracy.

	copied from https://github.com/Mattdl/CLsurvey/blob/master/src/methods/Finetune/train_SGD.py
	"""
	decayed_lr, stop_training = False, False

	# Early Stopping
	if count == early_stopping_patience:
		stop_training = True
		if is_master:
			print(f"Training terminated (after {time:.2f}s) as the val. loss has not improved in {early_stopping_patience} evaluations.")

	# LR Decay
	if count == decay_patience:
		decayed_lr = True
		for param_group in optimizer.param_groups:
			param_group['lr'] *= decay_factor
		if is_master:
			print(f'LR set to {optimizer.param_groups[0]["lr"]} as the val. loss has not improved in {decay_patience} evaluations.')

	return optimizer, decayed_lr, stop_training


def get_save_path(args, tag=None, fname=None, create=True, exp_id=None, run_id=None):
	exp_id_ = exp_id if exp_id is not None else args.exp_id
	run_id_ = run_id if run_id is not None else args.run_id
	save_path = Path(args.result_root) / args.dataset.lower() / exp_id_ / run_id_
	if tag is not None:
		save_path /= tag
	if create:
		save_path.mkdir(exist_ok=True, parents=True)
	if fname is not None:
		save_path /= fname
	return save_path


def save_checkpoint(model, args, tag, data=None, verbose=True):
	# obtain model state dictionary
	model_state_dict = model.state_dict()
	if isinstance(model, DDP):
		model_state_dict = {
			k.replace('module.', ''): v
			for k, v in model_state_dict.items()
		}
	if data is None:
		data = {}
	data['model'] = model_state_dict

	# save checkpoint
	ckpt_path = get_save_path(args, tag=tag, fname='model.ckpt')
	torch.save(data, ckpt_path)

	if verbose:
		print(f'Saved checkpoint @ {ckpt_path}.')


def load_checkpoint(model, args, tag, exp_id=None, run_id=None):
	# load checkpoint
	ckpt_path = get_save_path(args, tag=tag, fname='model.ckpt', create=False, exp_id=exp_id, run_id=run_id)
	if not ckpt_path.exists():
		raise RuntimeError(f"Checkpoint @ {ckpt_path} is required but doesn't exist!")
	data = torch.load(ckpt_path)

	# load checkpoint parameters into model
	model_state_dict = data['model']
	if isinstance(model, DDP):
		model_state_dict = {
			f"module.{k}": v
			for k, v in model_state_dict.items()
		}
	model.load_state_dict(model_state_dict)
	print(f'Loaded checkpoint from {ckpt_path}.')


def save_results(args, tag, results=None):
	# save result metrics
	if results is None:
		results = {}
	res_path = get_save_path(args, tag=tag, fname='results.pkl')
	with open(res_path, 'wb') as file:
		pickle.dump(results, file)
	print(f'Saved results @ {res_path}.')


def parameters_to_vector(model):
	return torch.cat([p.reshape(-1) for p in model.parameters()])


def parameters_to_vector_filter(model):
	all_p = []
	for n, p in model.named_parameters():
		if not ('downsample' in n and 'conv.weight' in n) and 'conv1.conv' not in n:
			all_p.append(p.reshape(-1))
	return torch.cat(all_p)
