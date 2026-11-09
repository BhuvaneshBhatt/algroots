"""POSIX resource budgets for isolated benchmark workers."""

import sys


def peak_rss_kib(usage):
    """Normalize macOS byte and other POSIX KiB peak-resident measurements."""
    return usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss


def resource_module():
    """Return POSIX process controls or report that this platform lacks them."""
    try:
        import resource
    except ModuleNotFoundError as error:
        raise NotImplementedError(
            "this benchmark worker requires POSIX process resource controls"
        ) from error
    return resource
