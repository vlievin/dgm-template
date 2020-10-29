import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.datasets import init_dataloaders
from lib.estimators import VariationalInference
from lib.training import get_loggers, summary2logger
from lib.training import init_model, init_optimizer, init_analyses
from lib.training import parse_arguments, parse_identifier, set_manual_seed
from lib.training import training_step, test_step
from lib.utils import Experiment, Session, ManualSeed
from lib.utils import Header, available_device, Aggregator, preprocess


def run():
    """Run the experiment"""

    # parse command line arguments
    args = parse_arguments()
    run_id = parse_identifier(args)

    with Header("Arguments"):
        for k, v in args.items():
            print(f"{k} : {v}")

    # initialize the Experiment object (logging directory + handling errors)
    with Experiment(args['root'], args['exp'], run_id, rf=args.get('rf', False)) as experiment:
        with Header("Experiment"):
            print(f"Run id: {run_id} \nPath: {os.path.abspath(experiment.logdir)} ")

        # get loggers
        logger_base, logger_train, logger_test = get_loggers(experiment.logdir, keys=['base', 'train', 'test'])

        # tensorboard writers used to log the summary
        writer_train = SummaryWriter(os.path.join(experiment.logdir, 'train'))
        writer_test = SummaryWriter(os.path.join(experiment.logdir, 'test'))

        # set random seed
        set_manual_seed(args.get('seed'))

        # initialize the data loaders
        loader_train, loader_valid, loader_test = init_dataloaders(args)

        # initialize the model
        model, hyperparameters = init_model(args, loader_train)

        # initialize the gradient estimator
        estimator = VariationalInference()

        # initialize the optimizer
        optimizer = init_optimizer(args, model, estimator)

        # define parameters (beta, ...)
        parameters = {}

        # training session (checkpointing), restore if a checkpoint exists
        session = Session(run_id, experiment.logdir, model, estimator, optimizer, hyperparameters)
        if session.restore_if_available():
            with Header("Session"):
                logger_base.info(f"Restoring Session from epoch = {session.epoch} (best test "
                                f"L = {session.best_elbo[0]:.3f} at step {session.best_elbo[1]}, "
                                f"epoch = {session.best_elbo[2]})")

        # move everything to device
        device = available_device()
        model = model.to(device)
        estimator = estimator.to(device)

        # define the sampler (display model samples)
        analyses = init_analyses(args, model=model, estimator=estimator, loader=loader_train,
                                 writer=writer_train, logger=logger_train,
                                 session=session, experiment=experiment)

        # main loop
        while session.epoch < args['epochs']:
            session.epoch += 1

            # test epoch
            if (session.epoch - 1) % args['eval_freq'] == 0:
                with torch.no_grad():
                    agg_test = Aggregator()
                    model.eval()
                    for batch in tqdm(loader_train, desc=f"[testing] "):
                        x, y = preprocess(batch, device)
                        diagnostics = test_step(x, model, estimator, y=y, **parameters)
                        agg_test.update(diagnostics)
                    summary_test = agg_test.data

                    # keep track of the best score and save the best model
                    session.save_best(summary_test['loss']['elbo'].mean().item())

                    # log the test summary
                    summary2logger(logger_test, summary_test, session.global_step, session.epoch, best=None)

                    # tensorboard logging
                    summary_test.log(writer_test, session.global_step)

                # analyses
                with ManualSeed(seed=args['seed']):
                    for analysis in analyses:
                        analysis()

            # training epoch
            agg_train = Aggregator()
            model.train()
            for batch in tqdm(loader_train, desc=f"[training] "):
                x, y = preprocess(batch, device)
                diagnostics = training_step(x, model, estimator, optimizer, y=y, **parameters)
                session.global_step += 1
                agg_train.update(diagnostics)
            summary_train = agg_train.data

            # log the training summary
            summary2logger(logger_train, summary_train, session.global_step, session.epoch, best=session.best_elbo)

            # tensorboard logging
            summary_train.log(writer_train, session.global_step)

            # checkpointing
            session.save()


if __name__ == '__main__':
    run()
