import os
import argparse
import numpy as np
import imageio as iio
from utils import load_mfa_model, to_im


def main(args):
    np.random.seed(seed=123)

    os.makedirs(args.out_folder, exist_ok=True)
    out_folder = os.path.join(args.out_folder, f'{args.dataset}_{args.attribute}')
    os.makedirs(out_folder, exist_ok=True)

    print("Initializing the MFA model...")
    model = load_mfa_model(args)

    print(f"Generating {args.n_samples} samples...")
    samples, labels = model.sample(args.n_samples)
    samples = to_im(samples, args.dataset)
    print(samples.shape, labels.shape)

    print(f"Writing images to {out_folder}...")
    for i, im in enumerate(samples):
        iio.imwrite(os.path.join(out_folder, f"{i:06}.jpeg"), samples[i])

    labels_file = os.path.join(args.out_folder, f'{args.dataset}_{args.attribute}.txt')
    print(f"Writing labels to {labels_file}...")
    with open(labels_file, "w") as f:
        for i, label in enumerate(labels):
            f.write(f"{i:06}.jpeg {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_folder', default='./models')
    parser.add_argument('--out_folder', default='./data')
    parser.add_argument('--dataset', choices=['celeba', 'mnist'], default='celeba')
    parser.add_argument('--attribute', default='Male')
    parser.add_argument('--dataset_type', choices=['original', 'symmetric', 'asymmetric'], default='symmetric')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--noise_std', type=float, default=0.01)

    main(parser.parse_args())
