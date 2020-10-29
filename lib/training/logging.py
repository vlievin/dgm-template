import logging
import os

def get_loggers(logdir, keys=['base', 'train', 'valid', 'test'],
                format='%(asctime)s %(name)-4s %(levelname)-4s %(message)s'):
    """get Logging loggers for a set of `keys`"""
    logging.basicConfig(level=logging.INFO,
                        format=format,
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.FileHandler(os.path.join(logdir, 'run.log')),
                                  logging.StreamHandler()])

    return (logging.getLogger(k) for k in keys)


def summary2logger(logger, summary, global_step, epoch, best=None, stats_key='loss', exp_id=''):
    """write summary to logging"""
    if not stats_key in summary.keys():
        logger.warning('key ' + str(stats_key) + ' not int output dictionary')
    else:
        message = f"{exp_id:32s}"
        message += f'[{global_step} / {epoch}]   '
        message += ''.join([f'{k} {v:6.2f}   ' for k, v in summary.get(stats_key).items()])
        if 'info' in summary.keys() and 'elapsed-time' in summary['info'].keys():
            message += f'({summary["info"]["elapsed-time"]:.2f}s /iter)'
        if best is not None:
            message += f'   (best: {best[0]:6.2f}  [{best[1]} / {best[2]}])'
        logger.info(message)
