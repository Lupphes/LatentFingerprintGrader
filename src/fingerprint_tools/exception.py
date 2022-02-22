import enum
import traceback
import sys


__all__ = [
    "ErrorCodes",

    "FingerToolsError",

    "ParseError",

    "FileError", "FileNotFoundError", "FileRestrictedError"
]


class ErrorCodes(enum.IntEnum):
    SUCCESS = 0,
    ERR_GENERAL_PARSE = 10,
    ERR_GENERAL_FILE = 20,
    ERR_OPENING_FILE = 21,
    ERR_WRITING_FILE = 22,
    ERR_INTERNAL = 999


class FingerToolsError(Exception):
    error_msg = "FingerTools encoutered an exception"
    exit_code = ErrorCodes.ERR_INTERNAL

    def __init__(self, *args):
        if args:
            self.error_msg = args[0]
        super().__init__(self.error_msg)
        self.log_exception()
        exit(self.exit_code)

    def log_exception(self):
        print(
            f"Encountered an error while runtime: {self.__class__.__name__}, {self.error_msg}")


class ParseError(FingerToolsError):
    error_msg = "The data specified couldn't be parsed."
    exit_code = ErrorCodes.ERR_GENERAL_PARSE

    def __init__(self, *args):
        super().__init__(*args)


class FileError(FingerToolsError):
    error_msg = "Encountered an exception while trying to interect with a specified file"
    exit_code = ErrorCodes.ERR_GENERAL_FILE

    def __init__(self, *args):
        super().__init__(*args)


class FileNotFoundError(FileError):
    error_msg = "Can't open file: No such file or directory"
    exit_code = ErrorCodes.ERR_OPENING_FILE

    def __init__(self, *args):
        super().__init__(*args)


class FileRestrictedError(FileError):
    error_msg = "Can't write into file: Permisson denied"
    exit_code = ErrorCodes.ERR_WRITING_FILE

    def __init__(self, *args):
        super().__init__(*args)
