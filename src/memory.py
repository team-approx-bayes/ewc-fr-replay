from typing import List
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


TYPE_MEMORABLE_PAST = 1
TYPE_ERROR_CORRECTION = 2


class PastTask:
    def __init__(self, class_ids, memorable_points, memorable_points_indices, memorable_points_types):
        self.class_ids = class_ids
        self.memorable_points = memorable_points
        self.memorable_points_indices = memorable_points_indices
        self.memorable_points_types = memorable_points_types
        self.mean = None

    @torch.no_grad()
    def update_mean(self, model, max_mem_per_batch=512):
        n_batches = int(np.ceil(len(self.memorable_points) / max_mem_per_batch))
        if n_batches == 1:
            self.mean = self._evaluate_mean(model).cpu()
        else:
            # Split forward passes into mini-batches to save memory
            mem_batch_indices = np.array_split(range(len(self.memorable_points)), n_batches)
            self.mean = torch.cat([self._evaluate_mean(model, idx=idx).cpu() for idx in mem_batch_indices])

    def _evaluate_mean(self, model, idx=None):
        memorable_points = self.memorable_points
        device = next(model.parameters()).device
        if idx is None:
            return model(memorable_points.to(device))
        else:
            return model(memorable_points[idx, :].to(device))


class Memory:
    def __init__(self,
                 model: torch.nn.Module,
                 temp=1.,
                 n_memorable_points=10,
                 memory_residual_frac=0.5,
                 use_experience_replay=False,
                 choose_m2_as_subset_of_m1=False,
                 ):
        self.model = model
        self.temp = temp
        self.n_memorable_points = n_memorable_points
        self.memory_residual_frac = memory_residual_frac
        self.use_experience_replay = use_experience_replay
        self.choose_m2_as_subset_of_m1 = choose_m2_as_subset_of_m1

        self.observed_tasks: List[PastTask] = []

    def update_memory_info(self,
                            data_loader: DataLoader,
                            dataset,
                            class_ids: List[int] = None,
                            is_distributed=False,
                            make_mem_train_loader=None):
        model = self.model
        if isinstance(model, DDP):
            # As DDP disables hook functions required for Kernel calculation,
            # the underlying module will be used instead.
            model = model.module
        model.eval()

        # register the current task with the memorable points
        with customize_head(model, class_ids):
            (memorable_points,
            memorable_points_indices,
            memorable_points_types) = collect_memorable_points(model,
                                                        data_loader,
                                                        dataset,
                                                        self.n_memorable_points,
                                                        self.memory_residual_frac,
                                                        self.use_experience_replay,
                                                        is_distributed,
                                                        make_mem_train_loader,
                                                        self.choose_m2_as_subset_of_m1)

        self.observed_tasks.append(PastTask(class_ids,
                                            memorable_points,
                                            memorable_points_indices,
                                            memorable_points_types))

        # update mean for each observed task
        for task in self.observed_tasks:
            with customize_head(model, task.class_ids, softmax=True, temp=self.temp):
                task.update_mean(model)


@torch.no_grad()
def collect_memorable_points(model,
                             data_loader: DataLoader,
                             dataset,
                             n_memorable_points,
                             memory_residual_frac=None,
                             use_experience_replay=False,
                             is_distributed=False,
                             make_mem_train_loader=None,
                             choose_m2_as_subset_of_m1=False):
    device = next(model.parameters()).device

    assert data_loader.batch_size is not None, 'DataLoader w/o batch_size is not supported.'
    assert not (use_experience_replay and memory_residual_frac is None)

    if is_distributed:
        indices = range(dist.get_rank(), len(dataset), dist.get_world_size())
        dataset = Subset(dataset, indices)

    assert not (choose_m2_as_subset_of_m1 and not use_experience_replay), 'Must use NN error correction for this option.'

    n_task_data = dataset.get_n_task_data() if getattr(dataset, 'get_n_task_data') else len(dataset)
    if use_experience_replay:
        n_error_correction_points = int(memory_residual_frac * n_memorable_points)
        n_memorable_points -= n_error_correction_points
    else:
        n_error_correction_points = 0
    n_points = {'memory': n_memorable_points, 'correction': n_error_correction_points}
    n_points_total = n_points["memory"] + n_points["correction"]

    assert n_points_total <= n_task_data,\
        f'# memory points ({n_points["memory"]} + {n_points["correction"]}) exceeds # data points ({n_task_data})!'

    memorable_points_indices = []
    memorable_points_types = []
    if choose_m2_as_subset_of_m1:
        memory_types = ['memory', 'correction']
    else:
        memory_types = (['correction'] if use_experience_replay else []) + ['memory']
    for memory_type in memory_types:
        n_points_= n_points_total if choose_m2_as_subset_of_m1 and memory_type == 'memory' else n_points[memory_type]
        if n_points_ == n_task_data - len(memorable_points_indices):
            print(f"Using all remaining {n_points_}/{n_task_data} data points as {memory_type} points.")
            memorable_points_indices += list(set(range(n_task_data)) - set(memorable_points_indices))
        else:
            if choose_m2_as_subset_of_m1 and memory_type == 'correction':
                subset_str = f' from the {n_points_total} memory points'
                exclude_indices = list(set(range(n_task_data)) - set(memorable_points_indices))
            else:
                subset_str = ''
                exclude_indices = memorable_points_indices

            print(f"Collecting {n_points_}/{n_task_data} {memory_type} points{subset_str}...")
            # compute random dataset scores
            dataset_scores = torch.tensor(np.random.rand(n_task_data))   # (n,)
            memorable_points_indices_new = _collect_memorable_points(
                                                       dataset_scores=dataset_scores,
                                                       n_memorable_points=n_points_,
                                                       exclude_indices=exclude_indices)
            if choose_m2_as_subset_of_m1 and memory_type == 'correction':
                memorable_points_indices = list(set(memorable_points_indices) - set(memorable_points_indices_new)) + memorable_points_indices_new
            else:
                memorable_points_indices += memorable_points_indices_new
        memory_type_id = TYPE_MEMORABLE_PAST if memory_type == 'memory' else TYPE_ERROR_CORRECTION
        memorable_points_types += [memory_type_id] * n_points[memory_type]

    # Convert within-task (i.e. starting at 0) to across-task memorable points indices
    global_mem_idx = lambda idx: dataset.globalize_memory_index(idx) 
    memorable_points_indices_global = [global_mem_idx(idx) for idx in memorable_points_indices]

    # create a Tensor for memorable points on model's device
    if make_mem_train_loader is None:
        memorable_points = [dataset[idx][0] for idx in memorable_points_indices_global]
        memorable_points = torch.stack(memorable_points).to(device)
    else:
        mem_train_loader = make_mem_train_loader(memorable_points_indices_global)
        memorable_points = torch.cat([batch[0] for batch in mem_train_loader]).float().to(device)
    return memorable_points, memorable_points_indices, memorable_points_types


def _collect_memorable_points(dataset_scores, n_memorable_points, exclude_indices=None):
    """ collect memorable points by sorting according to the score """

    # exclude indices
    indices = list(range(len(dataset_scores)))
    if exclude_indices is not None and len(exclude_indices) > 0:
        for idx in exclude_indices:
            if idx in indices.copy():
                indices.remove(idx)
    indices = np.array(indices)
    scores = dataset_scores[indices].numpy()

    indices_sorted = np.argsort(scores)[-n_memorable_points:]
    select_indices = torch.tensor(indices[indices_sorted])

    return select_indices.tolist()


@contextmanager
def customize_head(module: torch.nn.Module, class_ids: List[int] = None, softmax=False, temp=1.):

    def forward_hook(module, input, output):
        output /= temp
        if class_ids is not None:
            output = output[:, class_ids]
        if softmax:
            return F.softmax(output, dim=1)
        else:
            return output

    handle = module.register_forward_hook(forward_hook)
    yield
    handle.remove()
    del forward_hook
