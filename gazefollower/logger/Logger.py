# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import logging
import threading


class Log:
    instance = None
    _lock = threading.Lock()

    @classmethod
    def init(cls, log_file_path):
        cls.instance = cls()
        cls.instance.logger = cls._create_logger(log_file_path)

    @staticmethod
    def _create_logger(log_file_path):
        logger = logging.getLogger('gaze_follower_logger')
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @classmethod
    def i(cls, message):
        cls._check_logger()
        cls.instance.logger.info(message)

    @classmethod
    def d(cls, message):
        cls._check_logger()
        cls.instance.logger.debug(message)

    @classmethod
    def w(cls, message):
        cls._check_logger()
        cls.instance.logger.warning(message)

    @classmethod
    def e(cls, message):
        cls._check_logger()
        cls.instance.logger.error(message)

    @classmethod
    def _check_logger(cls):
        if cls.instance is None or cls.instance.logger is None:
            raise Exception("Logger has not been initialized. Please call Log.init() first.")
