""" Adapted from https://github.com/Mattdl/CLsurvey/blob/master/src/data/imgfolder.py """

import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
from torchvision import datasets

from memory import TYPE_MEMORABLE_PAST, TYPE_ERROR_CORRECTION
from utils import DATASET_IMAGENET

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

TYPE_NEW_DATA = 0


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}
	return classes, class_to_idx


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
	images = []
	dir = os.path.expanduser(dir)
	set_files = [line.rstrip('\n') for line in open(file_list)]
	for target in sorted(os.listdir(dir)):
		d = os.path.join(dir, target)
		if not os.path.isdir(d):
			continue

		for root, _, fnames in sorted(os.walk(d)):
			for fname in sorted(fnames):
				if is_image_file(fname):
					dir_file = target + '/' + fname
					if dir_file in set_files:
						path = os.path.join(root, fname)
						item = (path, class_to_idx[target])
						images.append(item)
	return images


class ImageFolderTrainVal(datasets.ImageFolder):
	def __init__(self, root, files_list, transform=None, target_transform=None,
				 loader=default_loader, classes=None, class_to_idx=None, imgs=None):
		"""
		:param root: root path of the dataset
		:param files_list: list of filenames to include in this dataset
		:param classes: classes to include, based on subdirs of root if None
		:param class_to_idx: overwrite class to idx mapping
		:param imgs: list of image paths (under root)
		"""
		if classes is None:
			assert class_to_idx is None
			classes, class_to_idx = find_classes(root)
		elif class_to_idx is None:
			class_to_idx = {classes[i]: i for i in range(len(classes))}
		print("Creating Imgfolder with root: {}".format(root))
		imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
		if len(imgs) == 0:
			raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
								format(root, ",".join(IMG_EXTENSIONS))))
		self.root = root
		self.samples = imgs
		self.classes = classes
		self.class_to_idx = class_to_idx
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader


class ImageFolder_Subset(datasets.ImageFolder):
	"""
	Wrapper of ImageFolder, subsetting based on indices.
	"""

	def __init__(self, dataset, indices):
		self.__dict__ = copy.deepcopy(dataset).__dict__
		self.indices = indices  # Extra

	def __getitem__(self, idx):
		return super().__getitem__(self.indices[idx])  # Only return from subset

	def __len__(self):
		return len(self.indices)


class ImageFolder_Concat(datasets.ImageFolder):
	"""
	Wrapper of ImageFolder that concatenates multiple datasets.
	"""

	def __init__(self, dataset_name, datasets=None, indices_per_task=None, class_ids_per_task=None, targets=None, n_train_data_all=None, gpu=None, convert_targets_multi_head=True):
		self.dataset_name = dataset_name

		if dataset_name == DATASET_IMAGENET:
			assert datasets is None, 'Cannot pass list of datasets to concatenate!'
			assert indices_per_task is not None, 'Need to pass list of data indices!'
			assert class_ids_per_task is not None, 'Need to pass list of class ids!'
			assert targets is not None, 'Need to pass targets of all data points!'
			assert n_train_data_all is not None, 'Need to pass number of all data points!'
			assert gpu is not None, 'Need to pass GPU id!'

			self.device = f'cuda:{gpu}'

			# Define task indices
			self.n_tasks = len(indices_per_task)
			self.indices_per_task = indices_per_task
			self.task_indices = [range(len(indices)) for indices in self.indices_per_task]
			self.update_task_indices()
			self.task_ids = torch.tensor(sum([[i] * len(d) for i, d in enumerate(self.indices_per_task)], [])).to(self.device)

			self.targets = targets.to(self.device)
			self.true_targets = self.targets.clone()
			self.data_types = torch.tensor([TYPE_NEW_DATA] * len(self)).to(self.device)

			self.n_train_data_all = n_train_data_all
			self.indices_ = torch.tensor(sum(self.indices_per_task, [])).to(self.device)
			self.update_local_indices()

			self.class_ids_per_task = torch.tensor(class_ids_per_task).to(self.device)
			self.use_soft_targets = False
			if convert_targets_multi_head:
				self.convert_targets_multi_head()

		else:
			assert datasets is not None, 'Need to pass list of datasets to concatenate!'
			assert indices_per_task is None, 'Cannot pass list of data indices!'
			assert class_ids_per_task is None, 'Cannot pass list of class ids!'
			assert targets is None, 'Cannot pass targets of all data points!'
			assert n_train_data_all is None, 'Cannot pass number of all data points!'
			assert gpu is None, 'Cannot pass GPU id!'

			# Take transforms and loader from the first dataset
			self.transform = datasets[0].transform
			self.target_transform = datasets[0].target_transform
			if hasattr(datasets[0], "loader"):  # only for (Tiny)ImageNet
				self.loader = datasets[0].loader
			self.use_soft_targets = False

			# Concatenate classes from all datasets
			if hasattr(datasets[0], "samples"):  # for (Tiny)ImageNet
				self.samples = sum([d.samples for d in datasets], [])
				self.true_targets = [s[1] for s in self.samples]
			elif hasattr(datasets[0], "data"):   # for SplitCIFAR
				self.data = np.concatenate([d.data for d in datasets])
				self.targets = sum([d.targets for d in datasets], [])
				self.true_targets = copy.deepcopy(self.targets)
			len_datasets = [len(d.samples) if hasattr(d, "samples") else len(d) for d in datasets]
			self.data_types = [TYPE_NEW_DATA] * sum(len_datasets)
			if hasattr(datasets[0], "class_to_idx"):
				self.class_ids_per_task = [torch.tensor(list(d.class_to_idx.values())) for d in datasets]
			else:
				self.class_ids_per_task = [d.class_ids_per_task[-1] for d in datasets]

			# Define task indices
			self.n_tasks = len(datasets)
			self.task_indices = [range(len(d)) for d in datasets]
			self.update_task_indices()
			self.task_ids = sum([[i] * l for i, l in enumerate(len_datasets)], [])

			if hasattr(datasets[0], "indices"):
				# If there are indices, also concatenate those
				self.indices = []
				offset = 0
				for i, d in enumerate(datasets):
					offset += 0 if i == 0 else len(datasets[i-1].samples)
					self.indices.append([idx + offset for idx in d.indices])
				self.indices = sum(self.indices, [])

			if convert_targets_multi_head:
				self.convert_targets_multi_head()

	def __getitem__(self, idx):
		idx_ = self._idx(idx)

		if self.dataset_name == DATASET_IMAGENET:
			data = (self.targets[idx_], self.true_targets[idx_])

		elif hasattr(self, "samples"):  # for (Tiny)ImageNet
			data = super().__getitem__(idx_) + (self.true_targets[idx_],)

		else:
			img, target, true_target = self.data[idx_], self.targets[idx_], self.true_targets[idx_]

			# doing this so that it is consistent with all other datasets
			# to return a PIL Image
			img = Image.fromarray(img)

			if self.transform is not None:
				img = self.transform(img)

			if self.target_transform is not None:
				target = self.target_transform(target)
				true_target = self.target_transform(true_target)

			data = (img, target, true_target)

		task_id = self.task_ids[idx_]
		data += (self.data_types[idx_], self.get_class_ids_per_task(task_id))
		data += (self.weights[task_id],) if hasattr(self, 'weights') else ()

		return data

	def __len__(self):
		if hasattr(self, "indices"):
			return len(self.indices)
		elif hasattr(self, "samples"):  # for (Tiny)ImageNet
			return super().__len__()
		else:   # for SplitCIFAR
			return len(self.targets)

	def _idx(self, idx):
		if hasattr(self, "indices"):
			return self.indices[idx]
		elif hasattr(self, "local_indices"):
			return self.local_indices[idx]
		else:
			return idx

	def get_all_class_ids(self):
		if isinstance(self.class_ids_per_task, torch.Tensor):
			return self.class_ids_per_task.tolist()
		else:
			return torch.cat(self.class_ids_per_task).tolist()

	def get_targets(self, data_ids):
		targets = self.targets[self._idx(data_ids)]
		if getattr(self, "target_transform ", None) is not None:
			targets = self.target_transform(targets)
		return targets

	def get_extra_data(self, data_ids):
		local_ids = self._idx(data_ids)
		task_ids = self.task_ids[local_ids]
		extra_data = (
			self.targets[local_ids],
			self.true_targets[local_ids],
			self.data_types[local_ids],
			self.class_ids_per_task[task_ids],
		)
		extra_data += (torch.stack(self.weights).to(self.device)[task_ids] if hasattr(self, 'weights') else None,)
		return extra_data

	def compute_avg_confidence(self, targets):
		return torch.max(targets, dim=1)[0].mean()

	def print_dataset_info(self):
		print(f"{len(self)} data points from {len(self.get_all_class_ids())} classes split across {len(self.class_ids_per_task)} task(s).")
		for t in range(self.n_tasks):
			class_ids = self.class_ids_per_task[t]
			data_ids = self.task_indices[t]
			if hasattr(self, 'taus'):
				tau = self.taus[t]
				tau_str = f', tau={tau:.6g}'
			else:
				tau_str = ''
			print(f"task {t+1} - class ids: {class_ids[0]}-{class_ids[-1]} ({len(class_ids)}), data ids: {data_ids[0]}-{data_ids[-1]} ({len(data_ids)}){tau_str} ", end='')

			task_targets = self.get_task_targets(t)
			if isinstance(task_targets[0], torch.Tensor):
				task_targets = torch.stack(task_targets, dim=0)
			else:
				task_targets = torch.tensor(task_targets)
			if self.use_soft_targets:
				if t < self.n_tasks-1:
					task_data_types = torch.tensor(self.get_task_data_types(t))
					avg_confidences = []
					for type_, name_ in [(TYPE_MEMORABLE_PAST, 'mem'), (TYPE_ERROR_CORRECTION, 'cor'), (TYPE_NEW_DATA, 'new')]:
						if (task_data_types == type_).any():
							avg_confidences.append((self.compute_avg_confidence(task_targets[task_data_types == type_]), name_))
				else:
					avg_confidences = [(self.compute_avg_confidence(task_targets), 'new')]
				conf_str = ', '.join([f'{conf:.3f} ({typ})' for conf, typ in avg_confidences])
				hard_soft_str = f"soft targets, avg. conf = {conf_str}"
			else:
				hard_soft_str = "hard targets"
			print(f"[{hard_soft_str}]")

	def set_taus_and_loss_weights(self, tau, tau2, n_task_data, scale_tau_by_task_size):
		if tau2 is None:
			tau2 = tau
		self.weights = []
		self.taus = []
		for t in range(self.n_tasks):
			if t == self.n_tasks - 1:
				tau_ = 1.0
			else:
				tau_ = tau if t == 0 else tau2
				if scale_tau_by_task_size:
					tau_ = tau_ * n_task_data[t] / self.get_n_task_data(t)
			self.taus.append(tau_)
			self.weights.append(torch.tensor([tau_] * len(self.class_ids_per_task[t])))

	def update_task_indices(self):
		offset = 0
		for i in range(len(self.task_indices)):
			offset += 0 if i == 0 else len(self.task_indices[i-1])
			self.task_indices[i] = [idx + offset for idx in range(len(self.task_indices[i]))]

	def update_local_indices(self):
		self.local_indices = torch.full(size=(self.n_train_data_all,), fill_value=self.n_train_data_all).to(self.device)
		self.local_indices[self.indices_] = torch.arange(len(self)).to(self.device)

	def set_memorable_points(self, task_id, mem_points_indices, mem_targets, mem_types):
		""" Set the (soft) targets for the memorable points of the given task 
			and remove all task points not part of the memorable past """

		assert self.use_soft_targets, 'Requires soft targets -- run convert_to_soft_targets first!'

		assert len(mem_targets) <= len(self.task_indices[task_id]),\
			f'There can be at most {len(self.task_indices[task_id])} < {len(mem_targets)} memorable_points!'

		assert mem_targets.shape[1] == len(self.class_ids_per_task[task_id]),\
			f'The targets need to cover {len(self.class_ids_per_task[task_id])} classes (not {mem_targets.shape[1]})!'

		assert (len(mem_targets) == len(mem_points_indices)) and (len(mem_targets) == len(mem_types)),\
			f'Memory data needs to be of same dimension ({len(mem_targets)}!={len(mem_points_indices)}!={len(mem_types)})!'

		for mem_idx, mem_target, mem_type in zip(mem_points_indices, mem_targets, mem_types):
			global_mem_idx = self.globalize_memory_index(mem_idx, task_id=task_id)
			idx_ = global_mem_idx if self.dataset_name == DATASET_IMAGENET else self._idx(global_mem_idx)
			if hasattr(self, "samples"):    # for (Tiny)ImageNet
				self.samples[idx_] = (self.samples[idx_][0], mem_target)
			else:   # for SplitCIFAR and FFCV-ImageNet
				self.targets[idx_] = mem_target
			self.data_types[idx_] = mem_type

		if self.dataset_name == DATASET_IMAGENET:
			self.indices_ = self.indices_.tolist()
			self.task_ids = self.task_ids.tolist()
			self.targets = self.targets.tolist()
			self.true_targets = self.true_targets.tolist()
			self.data_types = self.data_types.tolist()

			# remove indices of points not kept as memorable past
			for idx in range(len(self.task_indices[task_id]))[::-1]:
				if idx not in mem_points_indices:
					self.indices_per_task[task_id].pop(idx)
					global_mem_idx = self.globalize_memory_index(idx, task_id=task_id)
					del self.indices_[global_mem_idx]
					del self.task_ids[global_mem_idx]
					del self.targets[global_mem_idx]
					del self.true_targets[global_mem_idx]
					del self.data_types[global_mem_idx]

			self.indices_ = torch.tensor(self.indices_).to(self.device)
			self.task_ids = torch.tensor(self.task_ids).to(self.device)
			self.targets = torch.tensor(self.targets).to(self.device)
			self.true_targets = torch.tensor(self.true_targets).to(self.device)
			self.data_types = torch.tensor(self.data_types).to(self.device)
			self.update_local_indices()

		else:
			if not hasattr(self, "indices"):
				self.indices = list(range(len(self)))

			# remove indices of points not kept as memorable past
			for idx in range(len(self.task_indices[task_id]))[::-1]:
				if idx not in mem_points_indices:
					global_mem_idx = self.globalize_memory_index(idx, task_id=task_id)
					del self.indices[global_mem_idx]

		self.task_indices[task_id] = self.task_indices[task_id][:len(mem_points_indices)]
		self.update_task_indices()

	def globalize_memory_index(self, memory_index, task_id=-1):
		""" Convert within-task (i.e. starting at 0) to across-task memorable points indices """
		idx_fun = lambda idx: self.task_indices[task_id][idx] if hasattr(self, 'task_indices') else idx
		return idx_fun(memory_index)

	def remove_all_but_non_last_memorable_past(self, last_mem_points_indices):
		""" Remove all previous task points except those not part of the last memorable past. """

		if self.dataset_name == DATASET_IMAGENET:
			self.indices_ = self.indices_.tolist()
			self.task_ids = self.task_ids.tolist()
			self.targets = self.targets.tolist()
			self.true_targets = self.true_targets.tolist()
			self.data_types = self.data_types.tolist()

			# remove indices of points not kept as memorable past
			for t in range(1, self.n_tasks+1)[::-1]:
				for idx in range(len(self.task_indices[t-1]))[::-1]:
					if t < self.n_tasks or (t == self.n_tasks and idx in last_mem_points_indices):
						self.indices_per_task[t-1].pop(idx)
						global_mem_idx = self.globalize_memory_index(idx, task_id=t-1)
						del self.indices_[global_mem_idx]
						del self.task_ids[global_mem_idx]
						del self.targets[global_mem_idx]
						del self.true_targets[global_mem_idx]
						del self.data_types[global_mem_idx]

			self.indices_ = torch.tensor(self.indices_).to(self.device)
			self.task_ids = torch.tensor(self.task_ids).to(self.device)
			self.targets = torch.tensor(self.targets).to(self.device)
			self.true_targets = torch.tensor(self.true_targets).to(self.device)
			self.data_types = torch.tensor(self.data_types).to(self.device)
			self.update_local_indices()

		else:
			if not hasattr(self, "indices"):
				self.indices = list(range(len(self)))

			for t in range(1, self.n_tasks+1)[::-1]:
				for idx in range(len(self.task_indices[t-1]))[::-1]:
					if t < self.n_tasks or (t == self.n_tasks and idx in last_mem_points_indices):
						global_mem_idx = self.globalize_memory_index(idx, task_id=t-1)
						del self.indices[global_mem_idx]

		self.task_indices = [self.task_indices[self.n_tasks-1][len(last_mem_points_indices):]]
		self.update_task_indices()
		self.n_tasks = 1

		self.convert_to_hard_targets()

		self.class_ids_per_task = [self.class_ids_per_task[-1]]
		if self.dataset_name == DATASET_IMAGENET:
			self.class_ids_per_task = torch.stack(self.class_ids_per_task).to(self.device)

	def get_class_ids_per_task(self, task_id):
		""" Return class_ids of the task_id'th task (for train data) or the only task (for test data) """
		task_id_ = task_id if len(self.class_ids_per_task) > task_id else -1
		return self.class_ids_per_task[task_id_]

	def get_n_task_data(self, task_id=-1):
		return len(self.get_task_indices(task_id))

	def get_task_indices(self, task_id=-1):
		return (self.indices_per_task if self.dataset_name == DATASET_IMAGENET else self.task_indices)[task_id]

	def get_task_targets(self, task_id=-1):
		""" Get the targets for the given task (as they are, either soft or hard) """

		if hasattr(self, "samples"):    # for (Tiny)ImageNet
			get_target = lambda idx: self.samples[idx][1]
		else:   # for SplitCIFAR
			get_target = lambda idx: self.targets[idx]
		return [get_target(self._idx(idx)) for idx in self.get_task_indices(task_id)]

	def get_task_data_types(self, task_id=-1):
		return [self.data_types[self._idx(idx)] for idx in self.get_task_indices(task_id)]

	def convert_targets_multi_head(self):
		""" Converts the targets for use in a multi-head setting
			E.g., if we have two tasks with two classes each: t1=[0,1], t2=[2,3],
			then it will convert the targets of task 2 as: 2->0, 3->1
		"""

		assert not self.use_soft_targets, 'Can only convert hard targets to multi-head!'

		if self.dataset_name == DATASET_IMAGENET:
			self.targets = self.targets.tolist()
			self.true_targets = self.true_targets.tolist()

		targets = [s[1] for s in self.samples] if hasattr(self, "samples") else self.targets
		assert targets == self.true_targets, 'Targets must equal true targets!'

		for idx in range(len(self)):
			idx_ = idx if self.dataset_name == DATASET_IMAGENET else self._idx(idx)
			class_ids = self.get_class_ids_per_task(self.task_ids[idx_])
			target_orig = self.samples[idx_][1] if hasattr(self, "samples") else self.targets[idx_]
			target_new = (class_ids == target_orig).nonzero(as_tuple=False).item()
			if hasattr(self, "samples"):    # for (Tiny)ImageNet
				self.samples[idx_] = (self.samples[idx_][0], target_new)
			else:   # for SplitCIFAR
				self.targets[idx_] = target_new
			self.true_targets[idx_] = target_new

		if self.dataset_name == DATASET_IMAGENET:
			self.targets = torch.tensor(self.targets).to(self.device)
			self.true_targets = torch.tensor(self.true_targets).to(self.device)

	def convert_to_soft_targets(self):
		""" Converts the hard targets (i.e. scalars) into soft targets (i.e. one-hot vectors) """

		if self.use_soft_targets:
			print("Already using soft targets -- skipping.")
			return
		self.use_soft_targets = True

		if self.dataset_name == DATASET_IMAGENET:
			self.targets = self.targets.tolist()
			self.true_targets = self.true_targets.tolist()

		targets = [s[1] for s in self.samples] if hasattr(self, "samples") else self.targets
		assert targets == self.true_targets, 'Targets must equal true targets!'

		for idx in range(len(self)):
			idx_ = idx if self.dataset_name == DATASET_IMAGENET else self._idx(idx)
			one_hot_target = torch.zeros(len(self.class_ids_per_task[self.task_ids[idx_]]))
			cls = self.samples[idx_][1] if hasattr(self, "samples") else self.targets[idx_]
			one_hot_target[cls] = 1.
			if hasattr(self, "samples"):    # for (Tiny)ImageNet
				self.samples[idx_] = (self.samples[idx_][0], one_hot_target)
			else:   # for SplitCIFAR and FFCV-ImageNet
				self.targets[idx_] = one_hot_target
			self.true_targets[idx_] = one_hot_target.clone()

		if self.dataset_name == DATASET_IMAGENET:
			self.targets = torch.stack(self.targets).to(self.device)
			self.true_targets = torch.stack(self.true_targets).to(self.device)

	def convert_to_hard_targets(self):
		""" Converts the soft targets (i.e. one-hot vectors) into hard targets (i.e. scalars) """

		if not self.use_soft_targets:
			print("Already using hard targets -- skipping.")
			return
		self.use_soft_targets = False

		if self.dataset_name == DATASET_IMAGENET:
			self.targets = self.get_hard_targets(self.targets)
			self.true_targets = self.get_hard_targets(self.true_targets)

		else:
			if hasattr(self, "samples"):    # for (Tiny)ImageNet
				for idx in range(len(self)):
					i = self._idx(idx)
					self.samples[i] = (self.samples[i][0], self.get_hard_targets(self.samples[i][1].unsqueeze(0))[0])
					self.true_targets[i] = self.get_hard_targets(self.true_targets[i].unsqueeze(0))[0]

			else:   # for SplitCIFAR
				self.targets = self.get_hard_targets(torch.stack(self.targets)).tolist()
				self.true_targets = self.get_hard_targets(torch.stack(self.true_targets)).tolist()

	def get_hard_targets(self, targets):
		""" Returns the hard targets (i.e. scalars) corresponding to the soft targets (i.e. one-hot vectors) """
		return targets.argmax(axis=1)

	def subsample(self, class_ids, convert_targets_multi_head=False):
		""" Subsample dataset to only contain images of the given class_ids. """

		assert not self.dataset_name == DATASET_IMAGENET, 'Cannot subsample ImageNetFFCV!'

		class_ids = sorted(class_ids)
		self.class_ids_per_task = [torch.tensor(class_ids)]

		if hasattr(self, "samples"):	# for (Tiny)ImageNet
			indices_to_keep = [idx for idx, s in enumerate(self.samples) if s[1] in class_ids]
			self.samples = [self.samples[idx] for idx in indices_to_keep]
		else:	# for SplitCIFAR
			indices_to_keep = [idx for idx, target in enumerate(self.targets) if target in class_ids]
			self.data = np.stack([self.data[idx] for idx in indices_to_keep])

		self.targets = [self.targets[idx] for idx in indices_to_keep]
		self.true_targets = [self.true_targets[idx] for idx in indices_to_keep]
		self.task_ids = [self.task_ids[idx] for idx in indices_to_keep]
		self.data_types = [self.data_types[idx] for idx in indices_to_keep]

		if convert_targets_multi_head:
			self.convert_targets_multi_head()

		return self


def random_split(dataset, lengths):
	"""
	Creates ImageFolder_Subset subsets from the dataset, by altering the indices.
	:param dataset:
	:param lengths:
	:return: array of ImageFolder_Subset objects
	"""
	assert sum(lengths) == len(dataset)
	indices = torch.randperm(sum(lengths))
	return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
			zip(accumulate(lengths), lengths)]


class LaplaceDataset(datasets.ImageFolder):
	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return self.dataset.__len__()

	def __getitem__(self, idx):
		return self.dataset.__getitem__(idx)[:2]
