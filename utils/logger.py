import os
import logging

# set up logger for the main coe
def setup_logging(dataset_name):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{dataset_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='w'),
            logging.StreamHandler()
        ]
    )
    print(f"Logging to {log_filepath}")
