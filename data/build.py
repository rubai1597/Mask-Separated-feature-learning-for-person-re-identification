from torch.utils.data import DataLoader

from .transforms import build_transforms
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .collate_batch import collate_fn


def make_data_loader(args):
	print("initializing data {}".format(args.dataset_name))

	dataset = init_dataset(args.dataset_name, root=args.dataset_root)

	train_loader = DataLoader(
		ImageDataset(dataset.train, build_transforms(args, is_train=True)),
		batch_size=args.batch_size,
		sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance),
		num_workers=args.num_workers,
		collate_fn=collate_fn,
		drop_last=True
	)
	val_loader = DataLoader(
		ImageDataset(dataset.query + dataset.gallery, build_transforms(args, is_train=False)),
		batch_size=args.batch_size // 2,
		shuffle=False,
		num_workers=args.num_workers,
		collate_fn=collate_fn
	)

	return train_loader, val_loader, len(dataset.query), len(train_loader), dataset.num_train_pids
