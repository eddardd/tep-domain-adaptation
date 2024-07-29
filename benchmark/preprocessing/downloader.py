import os
import requests
from tqdm import tqdm
from requests.exceptions import RequestException
from benchmark.preprocessing import tep_data_urls


def tep_data_downloader(mode,
                        destination_path='./tep_data/benchmark/raw',
                        chunk_size=1024):
    """A simple downloader for getting the raw data of the
    Tennessee Eastman (TE) process.

    Parameters
    mode : integer
        An integer in the range 1, ..., 6, specifying which mode
        will be downloaded.
    destination_path : str
        String containing the path to the directory where data
        will be saved.
    chunk_size : int
        Integer specifying how many bits are processed per request
    """
    # Asserts that the mode is within the correct range
    assert mode in [1, 2, 3, 4, 5, 6], (
        f"Expected 'mode' to be in [1, ..., 6], but got {mode}"
    )

    # If folder does not exists, creates it
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # gets url and filename from dictionary
    url, filename = tep_data_urls[mode]

    # Gets complete destionation path
    destination = os.path.join(destination_path, filename)

    # Check if destination already exists
    if os.path.exists(destination):
        downloaded_size = os.path.getsize(destination)
    else:
        downloaded_size = 0

    # Set up starting point for download
    headers = {
        "Range": f"bytes={downloaded_size}-"
    }

    try:
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()

            # Gets total size of file being downloaded (useful for TQDM)
            total_size = int(
                response.headers.get('content-length', 0)) + downloaded_size

            # Writes content at destionation
            with open(destination, "ab") as file, tqdm(
                total=total_size,
                initial=downloaded_size,
                unit='B',
                unit_scale=True,
                desc=destination
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
                        pbar.update(len(chunk))
        print(f"\nDownload completed: {destination}")
    except RequestException as e:
        print(f"Download failed: {e}")
        print("Retrying...")
