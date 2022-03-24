import enum
import traceback
import sys


__all__ = [
    "ErrorCodes",

    "FingerToolsError",

    "ParseError",

    "FileError", "FileNotFoundError", "FileRestrictedError"

    "ArrgumentError"

    "UndefinedVariableError"

    "FingerprintQualityError"
]


class ErrorCodes(enum.IntEnum):
    SUCCESS = 0,
    ERR_GENERAL_PARSE = 10,
    ERR_GENERAL_FILE = 20,
    ERR_OPENING_FILE = 21,
    ERR_WRITING_FILE = 22,
    ERR_ARGUMENTS = 30,
    ERR_UNDEFINED_VARIABLE = 40,
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
    error_msg = "Encountered an exception while trying to interect with a specified file."
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


class ArrgumentError(FingerToolsError):
    error_msg = "The specified argument incorrect or missing. Please check your input."
    exit_code = ErrorCodes.ERR_WRITING_FILE

    def __init__(self, *args):
        super().__init__(*args)


class UndefinedVariableError(FingerToolsError):
    error_msg = "Variable which was called was not defined."
    exit_code = ErrorCodes.ERR_UNDEFINED_VARIABLE

    def __init__(self, *args):
        super().__init__(*args)


class FingerprintQualityError(FingerToolsError):
    error_msg = "Encountered a general exception while grading the fingerprint. Please check your input."
    exit_code = ErrorCodes.ERR_GENERAL_PARSE

    def __init__(self, *args):
        super().__init__(*args)


class EmptyMinutiaePointListError(FingerprintQualityError):
    error_msg = "Fingerprint has no minutiae points, which was not expected—error in analysis."
    exit_code = ErrorCodes.ERR_GENERAL_PARSE

    def __init__(self, *args):
        super().__init__(*args)


class NoFingerprintError(FingerprintQualityError):
    error_msg = "The fingerprint was not found on the image, which the script did not expect in the analysis."
    exit_code = ErrorCodes.ERR_GENERAL_PARSE

    def __init__(self, *args):
        super().__init__(*args)


class EmptyMaskError(FingerprintQualityError):
    error_msg = "Mask does not contain information about fingerprint position. It is empty."
    exit_code = ErrorCodes.ERR_GENERAL_PARSE

    def __init__(self, *args):
        super().__init__(*args)
