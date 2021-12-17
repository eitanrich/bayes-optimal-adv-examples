import os
import numpy as np
import mfa


class MFAClassifier:
    """
    A multi-class model where each class is modeled using a Mixture of Factor Analyzers
    """
    def __init__(self,
                 pretrained_model_folder,
                 n_classes=2,
                 force_sigma=None,
                 force_uniform_mixture=False,
                 ):
        self.class_models = {}
        self.n_classes = n_classes
        pi_eps = 1e-10

        print('Initializing MFAClassifier')
        for i in range(n_classes):
            print('Initializing class', i)
            mfa_model = mfa.MFA()
            mfa_model.load(os.path.join(pretrained_model_folder, 'mfa_class_{}'.format(i)))

            if force_sigma is not None:
                print('Forcing sigma to', force_sigma)
                for j, c in mfa_model.components.items():
                    c['D'][:] = force_sigma

            if force_uniform_mixture:
                num_non_padding_comps = len([1 for c in mfa_model.components.values() if c['pi'] > pi_eps])
                print('Forcing uniform mixture probabilities for {}/{} components.'.format(
                    num_non_padding_comps, len(mfa_model.components)))
                for j, c in mfa_model.components.items():
                    if c['pi'] > pi_eps:
                        c['pi'] = 1.0 / num_non_padding_comps

            self.class_models[i] = mfa_model
            noise_std = np.mean([np.mean(c['D']) for c in mfa_model.components.values()])
            print(f'Class {i}: {len(mfa_model.components)} components with mean noise std: {noise_std}')

    def sample(self, num_samples, add_noise=True):
        labels = np.random.randint(0, high=self.n_classes, size=num_samples)
        samples = np.empty([num_samples] + list(self.class_models[0].components[0]['mu'].shape), dtype=np.float32)
        for i in range(self.n_classes):
            num_class_samples = np.count_nonzero(labels == i)
            if num_class_samples > 0:
                samples[labels == i] = self.class_models[i].draw_samples(num_class_samples, add_noise=add_noise)
        return samples, labels
