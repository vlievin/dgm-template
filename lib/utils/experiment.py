import os
import signal
import sys
import traceback
import warnings
from shutil import rmtree

from .utils import logging_sep


class Experiment():
    """
    A class to handle experiments by creating the logging directory (if it doesn't exist)
    and writing the ´success´to a text file,
    So now you know why it has crashed miserably!

    Example:
    ´´´´python
    with Experiment(root, exp, run_id, rf) as experiment:
        ...
        # will write the exception to success.txt if an error is trown
    ´´´

    """

    file = 'success.txt'
    success = f"Success."
    aborted_by_user = f"Aborted by User."
    failure_base = "Failed."
    sigterm = "SIGTERM."

    def failure(exception):
        return f"{Experiment.failure_base} Exception : \n{exception}\n\n{traceback.format_exc()}"

    def __init__(self, root: str, exp: str, run_id: str, rf: bool = False):
        signal.signal(signal.SIGTERM, lambda *args: self.__exit__(*Experiment.sigterm_handler(*args)))

        # initialize the logging directory
        self.logdir = os.path.join(root, exp, run_id)
        self.init_logging_directory(rf)

    def init_logging_directory(self, rf: bool) -> None:
        """initialize the directory where will be saved the config, model's parameters and tensorboard logs"""
        if os.path.exists(self.logdir):
            if rf:
                warnings.warn(f"Deleting experiment directory ´{self.logdir}´")
                rmtree(self.logdir)
                os.makedirs(self.logdir)
            else:
                raise ValueError(f"Experiment directory ´{self.logdir}´ already exists. "
                                 f"Use the command argument ´--rf´ to override it.")
        else:
            os.makedirs(self.logdir)

    def __enter__(self):
        return self

    @staticmethod
    def sigterm_handler(_signo, _stack_frame):
        return (Experiment.sigterm, Experiment.sigterm, _stack_frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # handle success message
        message = {
            None: Experiment.success,
            KeyboardInterrupt: Experiment.aborted_by_user,
            Experiment.sigterm: Experiment.aborted_by_user,
        }.get(exc_type, None)

        if message is not None:
            # if message is successfully handled
            print(f"{logging_sep('=')}\n@ {sys.argv[0]} : {message}\n{logging_sep('=')}")
            if self.logdir is not None:
                with open(os.path.join(self.logdir, Experiment.file), 'w') as f:
                    f.write(message)
        else:
            # otherwise write the raw exception
            print(
                f"{logging_sep('=')}\n@ {sys.argv[0]}: Failed with exception {exc_type} = `{exc_val}` \n{logging_sep('=')}")
            traceback.print_exception(exc_type, exc_val, exc_tb)
            with open(os.path.join(self.logdir, Experiment.file), 'w') as f:
                f.write(Experiment.failure(exc_val))

        if exc_type == Experiment.sigterm:
            exit(0)
