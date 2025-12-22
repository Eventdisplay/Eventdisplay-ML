"""Utility functions for Eventdisplay-ML."""


def read_input_file_list(input_file_list):
    """
    Read a list of input files from a text file.

    Parameters
    ----------
    input_file_list : str
        Path to the text file containing the list of input files.

    Returns
    -------
    list of str
        List of input file paths.
    """
    try:
        with open(input_file_list) as f:
            input_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Error: Input file list not found: {input_file_list}") from exc

    return input_files
