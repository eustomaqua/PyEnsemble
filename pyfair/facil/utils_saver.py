# coding: utf-8
# Author: Yijun


import logging
import pandas as pd


# elegant print
# -----------------------


def _elegant_helper(text, logger=None):
    if not logger:
        print(text)
        return
    logger.info(text)


def elegant_print(text, logger=None):
    if isinstance(text, (
            str, pd.core.series.Series)):
        _elegant_helper(text, logger)
        return

    if not isinstance(text, list):
        raise ValueError("Wrong text input.")
    for k in text:
        _elegant_helper(k, logger)


# elegant logger
# -----------------------


def get_elogger(logname, filename, level=logging.DEBUG,
                mode='a'):
    logger = logging.getLogger(logname)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s | %(message)s",
        datefmt="%a %Y/%m/%d %H:%M:%S")  # "%D %H:%M:%S.%f"

    fileHandler = logging.FileHandler(filename, mode=mode)
    logger.setLevel(level)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # "%(asctime)s - %(levelname)s %(name)s | %(message)s",
    # fileHandler = logging.FileHandler(filename, mode='a')
    # return logger
    return logger, formatter, fileHandler


def rm_ehandler(logger, formatter, fileHandler):
    logger.removeHandler(fileHandler)
    del fileHandler, formatter
    del logger
    return


def console_out(log_document):
    # Output log to file and console

    # Define a Handler and set a format which output to file
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(name)s:%(levelname)s | %(message)s",
        format="%(message)s",  # "%(name)s:%(levelname)s| %(message)s",
        datefmt="%Y-%m-%d %A %H:%M:%S",
        filename=log_document,
        filemode='w')


# -----------------------
