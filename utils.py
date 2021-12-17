import os
import numpy as np
from mfa_classifier import MFAClassifier


def load_mfa_model(args):
    model_folder = os.path.join(args.models_folder, args.dataset, args.attribute)
    assert args.dataset_type == 'symmetric'

    # Load and prepare the model to be attacked
    return MFAClassifier(model_folder, force_sigma=args.noise_std, force_uniform_mixture=True)


def unflatten_samples(samples, dataset):
    if dataset == 'celeba':
        return samples.reshape([samples.shape[0], 64, 64, 3])
    else:
        return samples.reshape([samples.shape[0], 28, 28, 1])


def to_im(samples, dataset):
    return np.round(unflatten_samples(samples, dataset).clip(0., 1.)*255).astype(np.uint8)
