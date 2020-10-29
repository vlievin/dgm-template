import argparse
import hashlib
from typing import Dict

# define arguments that should not influence the run id
BASE_ARGS_EXCEPTIONS = ["root", "data_root", "exp", "epochs", "rf", "eval_freq", "analysis", "grad_samples"]

# arguments that will be explicitely written in the id
EXPLICIT_ARGS = ["dataset", "model", "seed"]


def parse_arguments() -> Dict:
    """parse command line arguments"""
    parser = argparse.ArgumentParser()

    # meta arguments
    parser.add_argument('--root', default='runs/',
                        help='directory to store training logs')
    parser.add_argument('--data_root', default='data/',
                        help='raw data directory')
    parser.add_argument('--exp', default='sandbox',
                        help='experiment directory: root/exp')
    parser.add_argument('--id', default='', type=str,
                        help='run id suffix')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='dataloader workers')
    parser.add_argument('--rf', action='store_true',
                        help='delete the existing experiment directory')

    # dataset arguments
    parser.add_argument('--dataset', default='binmnist',
                        help='dataset identifier')

    # training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='validation frequency')

    # optimization arguments
    parser.add_argument('--optimizer', default="adam", type=str,
                        help='optimizer identifier')
    parser.add_argument('--lr', default=2e-3, type=float,
                        help='learning rate')
    parser.add_argument('--grad_clip', default=1e18, type=float,
                        help='gradient clipping value')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='training batch-size')
    parser.add_argument('--eval_batch_size', default=128, type=int,
                        help='evaluation batch-size')

    # analysis
    parser.add_argument('--analysis', default="prior_sampling_image,gradient", type=str,
                        help='comma separated list of analyses (prior sampling, gradient stats,...)')
    parser.add_argument('--grad_samples', default=100, type=int,
                        help='number of Monte Carlo samples for the gradients')
    parser.add_argument('--grad_key_filter', default='', type=str,
                        help='comma separated list of patterns matching parameters names for the gradient analysis')

    # model arguments
    parser.add_argument('--model', default='vae', type=str,
                        help='model identifier')
    parser.add_argument('--hidden_size', default=64, type=int,
                        help='number of hidden units for each layer')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='number of hidden layers in each MLP')
    parser.add_argument('--dropout', default=0, type=float,
                        help='dropout rate')
    parser.add_argument('--num_latents', default=16, type=int,
                        help='latent space dimension')

    return vars(parser.parse_args())


def get_hash_from_args(args: Dict, exceptions=None):
    if exceptions is None:
        exceptions = BASE_ARGS_EXCEPTIONS
    filtered_opt_dict = {k: v for k, v in args.items() if k not in exceptions}
    opt_string = ",".join(("{}={}".format(*i) for i in filtered_opt_dict.items()))
    return hashlib.md5(opt_string.encode('utf-8')).hexdigest()


def parse_identifier(args: Dict) -> str:
    """parse the arguments and return a unique identifier"""
    hash = get_hash_from_args(args)
    base_id = "-".join(f"{args[k]}" for k in EXPLICIT_ARGS)
    id_suffix = f"{args['id']}-" if args['id'] != "" else ""
    return f"{id_suffix}{base_id}-{hash}"
