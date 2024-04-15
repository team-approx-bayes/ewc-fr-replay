import os
import copy

import numpy as np
import torch

from data.imgfolder import ImageFolder_Concat, ImageFolder_Subset
from utils import get_data_loader, get_dataset


class TaskFFCV:
    def __init__(self, task_id, class_ids, class_ids_test, train_loader, test_loader, decoder, train_dataset, test_dataset):
        self.id = task_id
        self.class_ids = class_ids
        self.class_ids_test = class_ids_test
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def clear_train_loader(self):
        self.train_loader = None


class Task:
    def __init__(self, task_id, class_ids, class_ids_test, train_set, test_set, batch_size, test_batch_size, n_workers, world_size, dataset_args, val_set=None, setup_loaders=False):
        self.id = task_id
        self.class_ids = class_ids
        self.class_ids_test = class_ids_test
        self._train_loader = None
        self._test_loader = None
        self._val_loader = None

        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.n_workers = n_workers
        self.world_size = world_size
        self.dataset_args = dataset_args

        if setup_loaders:
            self._setup_loaders()

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._setup_loaders()
        return self._train_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._setup_loaders()
        return self._test_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._setup_loaders()
        return self._val_loader

    @property
    def train_dataset(self):
        return self.train_loader.dataset

    @property
    def test_dataset(self):
        return self.test_loader.dataset

    @property
    def val_dataset(self):
        return self.val_loader.dataset

    def clear_train_loader(self):
        self._train_loader = None

    def _setup_loaders(self):
        self._train_loader, self._test_loader = get_data_loader(
            train_dataset=self.train_set,
            test_dataset=self.test_set,
            batch_size=self.batch_size,
            test_batch_size=self.test_batch_size,
            n_workers=self.n_workers,
            world_size=self.world_size,
            **self.dataset_args)

        if self.val_set is not None:
            _, self._val_loader = get_data_loader(
                train_dataset=self.train_set,
                test_dataset=self.val_set,
                batch_size=self.batch_size,
                test_batch_size=self.test_batch_size,
                n_workers=self.n_workers,
                world_size=self.world_size,
                **self.dataset_args)


def get_all_class_ids(test_set, args):
    test_targets = test_set.targets
    if hasattr(test_set, 'indices'):
        test_targets = torch.tensor(test_set.targets)[test_set.indices]
    if isinstance(test_targets, torch.Tensor):
        test_targets = test_targets.tolist()

    all_class_ids = np.sort(np.unique(test_targets))[:args.n_classes]
    assert len(all_class_ids) % args.n_tasks == 0

    return all_class_ids


def concat_datasets(args):
    return args.batch_mode == 'joint' or args.use_func_regularizer


def define_splitcifar_task_list(args, dataset_args, task_args, is_master=False):
    """ Define sequence of tasks for SplitCIFAR benchmark
        Task 1:    full CIFAR-10 dataset
        Tasks 2-6: 10 consecutive classes of CIFAR-100 each """

    # Apply random cropping and horizontal flipping
    dataset_args['random_crop'] = True
    dataset_args['random_crop_args'] = {'size': 32, 'padding': 4}
    dataset_args['random_horizontal_flip'] = True

    # Load CIFAR-10 dataset (for first task)
    dataset_args['dataset'] = "CIFAR-10"
    train_set_cifar10, test_set_cifar10 = get_dataset(**dataset_args)

    # Load CIFAR-100 dataset (for tasks 2-6) and shift targets by 10
    dataset_args['dataset'] = "CIFAR-100"
    train_set_cifar100, test_set_cifar100 = get_dataset(**dataset_args)
    train_set_cifar100.targets = [t + 10 for t in train_set_cifar100.targets]
    test_set_cifar100.targets = [t + 10 for t in test_set_cifar100.targets]
    train_set_cifar100.class_to_idx = {cls: t + 10 for cls, t in train_set_cifar100.class_to_idx.items()}
    test_set_cifar100.class_to_idx = {cls: t + 10 for cls, t in test_set_cifar100.class_to_idx.items()}

    # Concatenate CIFAR-10 and CIFAR-100 datasets
    train_set = ImageFolder_Concat(dataset_name=args.dataset, datasets=[train_set_cifar10, train_set_cifar100], convert_targets_multi_head=False)
    test_set = ImageFolder_Concat(dataset_name=args.dataset, datasets=[test_set_cifar10, test_set_cifar100], convert_targets_multi_head=False)

    all_class_ids = get_all_class_ids(test_set, args)

    task_list = []
    train_datasets = []
    for i, _class_ids in enumerate(np.array_split(all_class_ids, args.n_tasks)):
        _class_ids = sorted(_class_ids.tolist())
        if i >= args.end_task_id:
            continue
        if is_master:
            print(f"Preparing task #{i + 1} with class_ids={_class_ids}...")

        # Remove images of unused classes from datasets
        train_set_sub = copy.deepcopy(train_set).subsample(_class_ids)
        test_set_sub = copy.deepcopy(test_set).subsample(_class_ids, convert_targets_multi_head=True)

        # Append datasets to dataset lists
        train_datasets.append(train_set_sub)

        # Concatenate datasets for batch mode or memory mini-batching
        train_set_ = ImageFolder_Concat(dataset_name=args.dataset, datasets=train_datasets if concat_datasets(args) else [train_set_sub])

        class_ids = train_set_.get_all_class_ids()
        task_args.update(dict(
            train_set=train_set_,
            test_set=test_set_sub,
        ))
        task_list.append(Task(i + 1, class_ids, _class_ids, **task_args))

    return task_list


def define_tinyimagenet_task_list(args, dataset_args, task_args, is_master=False):
    """ define sequence of tasks for the SplitTinyImageNet benchmark """

    assert args.n_tasks == 10, 'SplitTinyImageNet must use 10 tasks.'

    # define data path
    tinyimgnet_path = os.path.join(args.dataset_root, 'tiny-imagenet-200')
    data_dir = os.path.join(tinyimgnet_path, 'no_crop', f'{args.n_tasks}tasks')

    # define dataset filenames
    from types import SimpleNamespace
    dset = SimpleNamespace()
    dset.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
    dset.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
    dataset_file = dset.transformed_dataset_file

    # load TinyImageNet dataset
    if dataset_args['download']:
        from data import tinyimgnet_dataprep
        tinyimgnet_dataprep.download_dset(args.dataset_root)
        tinyimgnet_dataprep.prepare_dataset(dset, tinyimgnet_path, task_count=args.n_tasks, survey_order=True, joint=False, overwrite=False)

    # define sequence of tasks
    n_classes_per_task = args.n_classes // args.n_tasks
    task_list = []
    datasets = {'train': [], 'test': [], 'val': []}
    for i in range(args.n_tasks):
        data_path = os.path.join(data_dir, f'{i+1}', dataset_file)
        dataset = torch.load(data_path)

        _class_ids = sorted(list(dataset['test'].class_to_idx.values()))
        if i >= args.end_task_id:
            continue
        if is_master:
            print(f"Preparing task #{i + 1} with class_ids={_class_ids}...")

        if n_classes_per_task < 20:
            # Subsample classes for all datasets
            dataset['test'] = ImageFolder_Subset(dataset['test'], list(range(len(dataset['test']))))
            for d in dataset.values():
                d.classes = d.classes[:n_classes_per_task]
                d.class_to_idx = {c: d.class_to_idx[c] for c in d.classes}
                d.indices = [i for i in d.indices if d.samples[i][1] in d.class_to_idx.values()]

        for d_name in dataset.keys():
            if args.data_frac[d_name] < 1.0:
                # Subsample dataset uniformly in class-balanced way
                n_data_per_class = int(args.n_data_per_class[d_name] * args.data_frac[d_name])
                all_indices = []
                for class_idx in dataset[d_name].class_to_idx.values():
                    class_indices = [i for i in dataset[d_name].indices if dataset[d_name].samples[i][1] == class_idx]
                    random_indices = np.random.permutation(range(len(class_indices)))[:n_data_per_class]
                    all_indices.append([class_indices[i] for i in random_indices])
                dataset[d_name].indices = sum(all_indices, [])

        # Append datasets to dataset lists
        for d in datasets:
            datasets[d].append(dataset[d])

        # Concatenate datasets for batch mode or memory mini-batching
        trainset = ImageFolder_Concat(dataset_name=args.dataset, datasets=datasets['train'] if concat_datasets(args) else [dataset['train']])
        valset = ImageFolder_Concat(dataset_name=args.dataset, datasets=datasets['val'] if concat_datasets(args) else [dataset['val']])
        testset = ImageFolder_Concat(dataset_name=args.dataset, datasets=[dataset['test']])

        # Extract list of of class IDs and define task
        class_ids = trainset.get_all_class_ids()
        task_args.update(dict(
            train_set=trainset,
            test_set=testset,
            val_set=valset,
            setup_loaders=True,
        ))
        task_list.append(Task(i + 1, class_ids, _class_ids, **task_args))

    return task_list


def define_imagenet_ffcv_task_list(args, dataset_args, task_args, gpu=0, is_master=False):
    """ define sequence of tasks for the ImageNet-1000 (FFCV) benchmark """

    from data.ffcv_utils import create_train_loader, create_val_loader

    assert args.n_classes == 1000 and args.n_tasks == 10, 'ImageNet-1000 requires 1000 classes across 10 tasks!'

    # define class and thus task sequence
    all_class_ids = list(range(args.n_classes))
    if args.shuffle_classes:
        prev_random_state = np.random.get_state()
        np.random.seed(args.shuffle_classes_seed)
        np.random.shuffle(all_class_ids)
        np.random.set_state(prev_random_state)
    class_ids_per_task = np.array_split(all_class_ids, args.n_tasks)
    class_id_to_task = {}
    for i, class_ids_ in enumerate(class_ids_per_task):
        for class_id in class_ids_:
            class_id_to_task[class_id] = i

    # Load targets and define mapping from class_id to list of corresponding data indices
    targets = {
        'train': torch.load('/home/edaxberger/data/imagenet_targets_train.pt'),
        'test': torch.load('/home/edaxberger/data/imagenet_targets_val.pt')}
    data_names = ['train', 'test']
    data_indices_per_task = {data_name: [[] for _ in range(args.n_tasks)] for data_name in data_names}
    for data_name in data_names:
        for data_index, target in enumerate(targets[data_name]):
            task_id = class_id_to_task[target.item()]
            data_indices_per_task[data_name][task_id].append(data_index)

    class_ids = {'train': [], 'test': []}
    data_indices = {'train': [], 'test': []}
    task_list = []
    for i, class_ids_ in enumerate(class_ids_per_task):
        class_ids['test'] = sorted(class_ids_.tolist())
        class_ids['train'] = (class_ids['train'] if i > 0 and concat_datasets(args) else []) + [class_ids['test']]
        if i >= args.end_task_id:
            continue
        if is_master:
            print(f"Preparing task #{i+1} with class_ids={class_ids['test']}...")

        # Identify data indices of this task's classes 
        data_indices['test'] = data_indices_per_task['test'][i]
        data_indices['train'] = (data_indices['train'] if i > 0 and concat_datasets(args) else []) + [data_indices_per_task['train'][i]] 
        train_data_indices = sum(data_indices['train'], [])

        # Create FFCV data loaders
        train_loader, decoder = create_train_loader(
            train_dataset=dataset_args['train_root'],
            num_workers=task_args['n_workers'],
            batch_size=task_args['batch_size'],
            in_memory=True,
            start_ramp=args.start_ramp,
            end_ramp=args.end_ramp,
            gpu=gpu,
            indices=train_data_indices)
        test_loader = create_val_loader(
            val_dataset=dataset_args['val_root'],
            num_workers=task_args['n_workers'],
            batch_size=task_args['test_batch_size'],
            gpu=gpu,
            indices=copy.deepcopy(data_indices['test']))

        train_dataset = ImageFolder_Concat(dataset_name=args.dataset,
                                           indices_per_task=copy.deepcopy(data_indices['train']),
                                           class_ids_per_task=copy.deepcopy(class_ids['train']),
                                           targets=targets['train'].clone()[train_data_indices],
                                           n_train_data_all=len(targets['train']),
                                           gpu=gpu)
        test_dataset = ImageFolder_Concat(dataset_name=args.dataset,
                                           indices_per_task=[copy.deepcopy(data_indices['test'])],
                                           class_ids_per_task=[copy.deepcopy(class_ids['test'])],
                                           targets=targets['test'].clone()[data_indices['test']],
                                           n_train_data_all=len(targets['test']),
                                           gpu=gpu)
        task_list.append(TaskFFCV(i + 1, sorted(sum(class_ids['train'], [])), copy.deepcopy(class_ids['test']),
                                  train_loader, test_loader, decoder, train_dataset, test_dataset))

    return task_list
