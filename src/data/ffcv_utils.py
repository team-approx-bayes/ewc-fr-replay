""" Adapted from https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py """

import torch as ch
import torch.nn.functional as F
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import numpy as np

from typing import List
from pathlib import Path

from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
	RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
	RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

Section('model', 'model details').params(
	arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
	pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
	min_res=Param(int, 'the minimum (starting) resolution', default=160),
	max_res=Param(int, 'the maximum (starting) resolution', default=160),
	end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
	start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
	train_dataset=Param(str, '.dat file to use for training', required=True),
	val_dataset=Param(str, '.dat file to use for validation', required=True),
	num_workers=Param(int, 'The number of workers', required=True),
	in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
	step_ratio=Param(float, 'learning rate step ratio', default=0.1),
	step_length=Param(int, 'learning rate step length', default=30),
	lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
	lr=Param(float, 'learning rate', default=0.5),
	lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
	folder=Param(str, 'log location', required=True),
	log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('validation', 'Validation parameters stuff').params(
	batch_size=Param(int, 'The batch size for validation', default=512),
	resolution=Param(int, 'final resized validation image size', default=224),
	lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
	eval_only=Param(int, 'eval only?', default=0),
	batch_size=Param(int, 'The batch size', default=512),
	optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
	momentum=Param(float, 'SGD momentum', default=0.9),
	weight_decay=Param(float, 'weight decay', default=4e-5),
	epochs=Param(int, 'number of epochs', default=30),
	label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
	distributed=Param(int, 'is distributed?', default=0),
	use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
	world_size=Param(int, 'number gpus', default=1),
	address=Param(str, 'address', default='localhost'),
	port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

start_end_ramps = {
	16: (11, 13),
	24: (17, 20),
	32: (23, 27),
	40: (29, 34),
	56: (41, 48),
	88: (65, 76),
}

class BlurPoolConv2d(ch.nn.Module):
	def __init__(self, conv):
		super().__init__()
		default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
		filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
		self.conv = conv
		self.register_buffer('blur_filter', filt)

	def forward(self, x):
		blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
						groups=self.conv.in_channels, bias=None)
		return self.conv.forward(blurred)


def create_optimizer(model, optimizer, momentum, weight_decay):
	assert optimizer.lower() == 'sgd'

	# Only do weight decay on non-batchnorm parameters
	all_params = list(model.named_parameters())
	bn_params = [v for k, v in all_params if ('bn' in k)]
	other_params = [v for k, v in all_params if not ('bn' in k)]
	param_groups = [{
		'params': bn_params,
		'weight_decay': 0.
	}, {
		'params': other_params,
		'weight_decay': weight_decay
	}]
	return ch.optim.SGD(param_groups, lr=1, momentum=momentum)


def get_lr(epoch, epochs, optim_args):
	lr_schedules = {
		'cyclic': get_cyclic_lr,
		'step': get_step_lr
	}
	return lr_schedules[optim_args['lr_schedule_type']](epoch, epochs, optim_args)


def get_step_lr(epoch, epochs, optim_args):
	if epoch >= epochs:
		return 0

	num_steps = epoch // optim_args['step_length']
	return optim_args['step_ratio']**num_steps * optim_args['lr']


def get_cyclic_lr(epoch, epochs, optim_args):
	xs = [0, optim_args['lr_peak_epoch'], epochs]
	ys = [1e-4 * optim_args['lr'], optim_args['lr'], 0]
	return np.interp([epoch], xs, ys)[0]


def get_resolution(epoch, start_ramp, end_ramp, min_res=160, max_res=192):
	assert min_res <= max_res

	if epoch <= start_ramp:
		return min_res

	if epoch >= end_ramp:
		return max_res

	# otherwise, linearly interpolate to the nearest multiple of 32
	interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
	final_res = int(np.round(interp[0] / 32)) * 32
	return final_res


def create_train_loader(train_dataset, num_workers, batch_size, in_memory, start_ramp, end_ramp, gpu, indices, drop_last=True, shuffle=True):
	this_device = f'cuda:{gpu}'
	train_path = Path(train_dataset)
	assert train_path.is_file()

	res = get_resolution(epoch=0, start_ramp=start_ramp, end_ramp=end_ramp)
	decoder = RandomResizedCropRGBImageDecoder((res, res))
	image_pipeline: List[Operation] = [
		decoder,
		RandomHorizontalFlip(),
		ToTensor(),
		ToDevice(ch.device(this_device), non_blocking=True),
		ToTorchImage(),
		NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
	]

	index_pipeline: List[Operation] = [
		IntDecoder(),
		ToTensor(),
		Squeeze(),
		ToDevice(ch.device(this_device), non_blocking=True)
	]

	loader = Loader(train_dataset,
					batch_size=batch_size,
					num_workers=num_workers,
					order=OrderOption.QUASI_RANDOM if shuffle else OrderOption.SEQUENTIAL,
					os_cache=in_memory,
					drop_last=drop_last,
					indices=indices,
					pipelines={
						'image': image_pipeline,
						'index': index_pipeline,
					})

	return loader, decoder


def create_val_loader(val_dataset, num_workers, batch_size, gpu, indices, resolution=256):
	this_device = f'cuda:{gpu}'
	val_path = Path(val_dataset)
	assert val_path.is_file()
	res_tuple = (resolution, resolution)
	cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
	image_pipeline = [
		cropper,
		ToTensor(),
		ToDevice(ch.device(this_device), non_blocking=True),
		ToTorchImage(),
		NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
	]

	index_pipeline = [
		IntDecoder(),
		ToTensor(),
		Squeeze(),
		ToDevice(ch.device(this_device), non_blocking=True)
	]

	loader = Loader(val_dataset,
					batch_size=batch_size,
					num_workers=num_workers,
					order=OrderOption.SEQUENTIAL,
					drop_last=False,
					indices=indices,
					pipelines={
						'image': image_pipeline,
						'index': index_pipeline,
					})
	return loader


class FFCVTargetDataLoader:
	""" This class defines an iterator that wraps an FFCV DataLoader 
		to return targets instead of data indices (for use with laplace).
	"""
	def __init__(self, ffcv_dataloader, dataset):
		self.ffcv_dataloader = ffcv_dataloader
		self.dataset = dataset

	def __iter__(self):
		self.data_iter = iter(self.ffcv_dataloader)
		return self

	def __next__(self):
		inputs, data_ids = next(self.data_iter)
		targets = self.dataset.get_targets(data_ids)
		return inputs.float(), targets

	def __len__(self):
		return len(self.ffcv_dataloader)
