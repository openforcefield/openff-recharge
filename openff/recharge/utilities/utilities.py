import errno
import os


def get_data_file_path(relative_path: str) -> str:
    """Get the full path to one of the files in the data directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.

    Returns
    -------
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
    """

    from pkg_resources import resource_filename

    file_path = resource_filename(
        "openff.recharge", os.path.join("data", relative_path)
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    return file_path
