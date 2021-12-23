import logging
import os

loggers = {}

def get_logger(module_name):

    # global loggers
    
    if module_name in loggers.keys():
        return loggers[module_name]
    
    else:
        # copied from https://realpython.com/python-logging/

        # Create a custom logger
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)

        # if to_logfile:
        #     fn_log = to_logfile
        #     f_handler = logging.FileHandler(fn_log)
        #     f_handler.setLevel(logging.INFO)

        #     # Create formatters and add it to handlers
        #     f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     f_handler.setFormatter(f_format)


        #     logger.addHandler(f_handler)
        
        # logger.propagate = False
        loggers[module_name] = logger
        return logger
        
