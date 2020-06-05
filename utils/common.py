import os
import math
import numpy as np
from argparse import ArgumentTypeError
from matplotlib import cm


def check_directory(arg, access=os.W_OK, access_str="writeable"):
	""" Check for directory-type argument validity.
	Checks whether the given `arg` commandline argument is either a readable
	existing directory, or a createable/writeable directory.
	Args:
		arg (string): The commandline argument to check.
		access (constant): What access rights to the directory are requested.
		access_str (string): Used for the error message.
	Returns:
		The string passed din `arg` if the checks succeed.
	Raises:
		ArgumentTypeError if the checks fail.
	"""
	path_head = arg
	while path_head:
		if os.path.exists(path_head):
			if os.access(path_head, access):
				# Seems legit, but it still doesn't guarantee a valid path.
				# We'll just go with it for now though.
				return arg
			else:
				raise ArgumentTypeError(
					'The provided string `{0}` is not a valid {1} path '
					'since {2} is an existing folder without {1} access.'
					''.format(arg, access_str, path_head))
		path_head, _ = os.path.split(path_head)

	# No part of the provided string exists and can be written on.
	raise ArgumentTypeError('The provided string `{}` is not a valid {}'
							' path.'.format(arg, access_str))


def writeable_directory(arg):
	""" To be used as a type for `ArgumentParser.add_argument`. """
	return check_directory(arg, os.W_OK, "writeable")


def readable_directory(arg):
	""" To be used as a type for `ArgumentParser.add_argument`. """
	return check_directory(arg, os.R_OK, "readable")


def number_greater_x(arg, type_, x):
	try:
		value = type_(arg)
	except ValueError:
		raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
			arg, type_.__name__))

	if value > x:
		return value
	else:
		raise ArgumentTypeError('Found {} where an {} greater than {} was '
			'required'.format(arg, type_.__name__, x))


def number_between_x(arg, type_, xmin, xmax):
	try:
		value = type_(arg)
	except ValueError:
		raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
			arg, type_.__name__))

	if xmin < value < xmax:
		return value
	else:
		raise ArgumentTypeError('Found {} where an {} is between {} and {} was '
			'required'.format(arg, type_.__name__, xmin, xmax))


def positive_int(arg):
	return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
	return number_greater_x(arg, int, -1)


def positive_float(arg):
	return number_greater_x(arg, float, 0)


def probability(args):
	return number_between_x(args, float, 0, 1)


class AverageMeter(object):
	def __init__(self):
		self.n = 0
		self.sum = 0.0
		self.var = 0.0
		self.val = 0.0
		self.mean = np.nan
		self.std = np.nan

	def update(self, value, n=1):
		self.val = value
		self.sum += value
		self.var += value * value
		self.n += n

		if self.n == 0:
			self.mean, self.std = np.nan, np.nan
		elif self.n == 1:
			self.mean, self.std = self.sum, np.inf
		else:
			self.mean = self.sum / self.n
			self.std = math.sqrt(
				abs(self.var - self.n * self.mean * self.mean) / (self.n - 1.0))

	def value(self):
		return self.mean, self.std

	def reset(self):
		self.n = 0
		self.sum = 0.0
		self.var = 0.0
		self.val = 0.0
		self.mean = np.nan
		self.std = np.nan
