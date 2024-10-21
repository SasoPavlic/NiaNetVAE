import json
import logging
import re
import sys
from colorama import Fore, Back, Style
from colorama import init as colorama_init

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, record):
        # Allow log records with level less than the max_level
        return record.levelno < self.max_level

class Log:
    """Class responsible for logging information."""

    HEADER_W = [Fore.BLACK, Back.WHITE, Style.BRIGHT]
    HEADER_R = [Fore.WHITE, Back.RED, Style.BRIGHT]
    HEADER_G = [Fore.WHITE, Back.GREEN, Style.BRIGHT]

    @classmethod
    def enable(cls, storage):
        colorama_init()
        cls.logger = logging.getLogger(storage['name'])
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.propagate = False  # Prevent propagation to ancestor loggers

        # Stream handler for stdout (DEBUG, INFO, WARNING)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_formatter = logging.Formatter("%(message)s")
        stdout_handler.setFormatter(stdout_formatter)
        stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))

        # Stream handler for stderr (ERROR, CRITICAL)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_formatter = logging.Formatter("%(message)s")
        stderr_handler.setFormatter(stderr_formatter)

        # Add handlers to logger
        cls.logger.addHandler(stdout_handler)
        cls.logger.addHandler(stderr_handler)

        # Create and setup file handler
        file_handler = logging.FileHandler(storage['save_dir'] + storage['logger_file'])
        file_formatter = FileFormatter("%(asctime)s\n%(message)s")
        file_handler.setFormatter(file_formatter)
        cls.logger.addHandler(file_handler)

    @classmethod
    def header(cls, message, type="WHITE"):
        options = cls.HEADER_W if type == "WHITE" else cls.HEADER_R if type == "RED" else cls.HEADER_G
        cls.info(message.center(80, '-'), options)

    @classmethod
    def debug(cls, message, options=[Fore.CYAN]):
        cls.logger.debug(cls.create_message(message, options))

    @classmethod
    def info(cls, message, options=[Fore.GREEN]):
        cls.logger.info(cls.create_message(message, options))

    @classmethod
    def warning(cls, message, options=[Fore.YELLOW]):
        cls.logger.warning(cls.create_message(message, options))

    @classmethod
    def error(cls, message, options=[Fore.MAGENTA]):
        cls.logger.error(cls.create_message(message, options))

    @classmethod
    def critical(cls, message, options=[Fore.RED, Style.BRIGHT]):
        cls.logger.critical(cls.create_message(message, options))

    @classmethod
    def create_message(cls, message, options):
        if isinstance(message, dict):
            message = json.dumps(message, indent=4, sort_keys=True)
        if not isinstance(message, str):
            message = str(message)
        return ''.join(options) + message + '\033[0m'

class FileFormatter(logging.Formatter):
    def plain(self, string):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]|[-]{2,}')
        return ansi_escape.sub('', string)

    def format(self, record):
        message = super().format(record)
        plain_message = self.plain(message)
        separator = '=' * 80
        return ''.join((separator, "\n", plain_message, "\n", separator))
