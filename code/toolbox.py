import time
import logging
import os
import fcntl

from typing import Callable, List, Any
from functools import partial
from multiprocessing import cpu_count, Event
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class FileHandlerWithLock(logging.FileHandler):
    """Custom FileHandler that uses a file lock to prevent concurrent writes."""
    def emit(self, record):
        with open(self.baseFilename, self.mode) as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                self.stream = f
                super().emit(record)
            finally:
                self.stream = None
                fcntl.flock(f, fcntl.LOCK_UN)  # Release lock

def get_logger(name: str, save_dir: str = ''):
    """Create a logger for the given name and save directory.
    Args:
        name (str): Name of the logger.
        save_dir (str): Directory to save the log file.
    Returns:
        logging.Logger: Configured logger object.
    """
    # Check if save_dir exists
    if save_dir and save_dir[-1] != '/':
        save_dir += '/'
    
    if save_dir and not os.path.exists(save_dir):
        # Use try for parallel creation of directories
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass

    # Initialize logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = FileHandlerWithLock(save_dir + name + '.log')
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def run_function(LOGGER: logging.Logger, target: Callable[..., Any], items: List[Any], Parallel: bool=True, P_type: str='thread', N_CPUS: int=0, stop_flag: Event=None, *args, **kwargs) -> List[Any]:
    """Run a function with a list of items in parallel or sequentially.
    Args:
        LOGGER (logging.Logger): Logger object for logging.
        target (Callable[..., Any]): The function to run.
        items (List[Any]): List of items to process.
        Parallel (bool): Whether to run in parallel or sequentially.
        P_type (str): Type of parallelism ('thread' or 'process').
        N_CPUS (int): Number of CPUs to use for parallel processing.
        *args: Additional arguments to pass to the target function.
        **kwargs: Additional keyword arguments to pass to the target function.
    Returns:
        List[Any]: List of results from the target function.
    """

    target_name = target.func.__name__ if isinstance(target, partial) else target.__name__
    if N_CPUS == 0:
        N_CPUS = cpu_count() - 1
    else:
        N_CPUS = min(N_CPUS, cpu_count() - 1)
            
    # Debugging information
    LOGGER.debug(f'Running {target_name} {" in parallel" if Parallel else "sequentially"}')
    LOGGER.debug(f'Number of items: {len(items)}')

    # Run the target function with a progress bar
    results = []
    try:
        if Parallel:
            max_workers = min(32, 2 * N_CPUS)
            LOGGER.debug(f'Using {P_type} with max_workers={max_workers}')
            Executor = ThreadPoolExecutor if P_type == 'thread' else ProcessPoolExecutor
            with Executor(max_workers=max_workers) as executor:
                futures = [executor.submit(target, item, *args, **kwargs) for item in items]
                for i, future in enumerate(futures):
                    if stop_flag and stop_flag.is_set():
                        LOGGER.info('Stopping parallel processing due to stop flag')
                        break
                    retries = 3
                    while retries > 0:
                        try:
                            LOGGER.debug(f'Waiting for future {i} to complete: {retries} retries left') 
                            result = future.result(timeout=300)
                            results.append(result)
                            LOGGER.debug(f'Future {i} completed successfully')
                            break
                        except TimeoutError:
                            LOGGER.error(f'Timeout error for item {i}. Retrying...')
                            retries -= 1
                        except KeyboardInterrupt:
                            LOGGER.error('KeyboardInterrupt received. Stopping processing.')
                            if stop_flag:
                                stop_flag.set()
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                        except Exception as e:
                            LOGGER.error(f'Error in parallel processing for item {i}: {e}', exc_info=True)
                            retries -= 1
                        if retries == 0:
                            LOGGER.error(f'Max retries reached for item {i}. Skipping...')
        else:
            for item in items:
                if stop_flag and stop_flag.is_set():
                    LOGGER.info('Stopping sequential processing due to stop flag')
                    break
                try:
                    result = target(item, *args, **kwargs)
                    results.append(result)
                except Exception as e:
                    LOGGER.error(f'Error in sequential processing: {e}')
    except KeyboardInterrupt:
        LOGGER.error('KeyboardInterrupt received. Stopping processing.')
        if stop_flag:
            stop_flag.set()
    finally:
        LOGGER.debug(f'Completed {target_name} {" in parallel" if Parallel else "sequentially"}')
        LOGGER.debug(f'Number of results: {len(results)}')

    # Check if results is a list of tuples before returning zip(*results)
    if results and isinstance(results[0], tuple):
        return zip(*results)
    return results

class ProgressBar:
    # Class to create a progress bar
    # Will display a progress bar with the current progress, the current step, the status, and the estimated time remaining
    def __init__(self, total, splits=20, update_interval=1):
        self.total = total
        self.splits = splits
        self.current = 0
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update(index=0)

    def update(self, index = None, status=''):
       # with self.lock:
        if index is None:
            index = self.current + 1

        if index % self.update_interval != 0 and index != self.total:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if index > 0:
            avg_time_per_step = elapsed_time / index
            remaining_steps = self.total - index
            eta = avg_time_per_step * remaining_steps
        else:
            eta = 0

        current = int((index / self.total) * self.splits)
        current_progress = ''
        for i in range(self.splits):
            if i < current:
                current_progress += '■'
            else:
                current_progress += '□'

        eta_formatted = self.format_time(eta)
        print(f'\r {current_progress} | {index}/{self.total} | {status} | ETA: {eta_formatted} |', end='', flush=True)
        self.current = index

    @staticmethod
    def format_time(seconds):
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        return f'{hours:02}:{mins:02}:{secs:02}'

# Example usage
if __name__ == '__main__':
    import random
    total_steps = 100
    progress_bar = ProgressBar(total_steps)

    for i in range(total_steps):
        time.sleep(random.random()/2)  # Simulate work
        progress_bar.update(i + 1, status='Processing')