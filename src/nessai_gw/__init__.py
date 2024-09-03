import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

nessai_logger = logging.getLogger("nessai")
# logger is called 'nessai.nessai_gw'
# This ensures the logging changes propagate to loggers in this package.
nessai_logger.getChild(__name__)
