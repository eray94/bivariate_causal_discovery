class Error(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message=""):
        self.message = message

class SizeMismatchError(Error):
    """Exception raised for errors if size of ground truth and predictions are not match."""

    def __init__(self, message=""):
        message = "SizeMismatchError " + message
        super().__init__(message=message)

class AucError(Error):
    """Exception raised for errors in the auc calculation."""

    def __init__(self, message=""):
        message = "AucError " + message
        super().__init__(message=message)
