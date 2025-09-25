# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import logging
import re
import os
import shutil
import subprocess
import urllib
from pathlib import Path
from multiprocessing.pool import ThreadPool
from itertools import repeat
import requests
from urllib import parse, request
from vajra.utils import LOGGER, clean_url, TQDM, is_online, url2file
from vajra import checks
import torch

REPO = "NamanMakkar/VayuAI"
GITHUB_ASSETS_NAMES = (
    [f'visdrone-best-vajra-v1-{k}-det.pt' for k in ('nano', 'small', 'medium', 'large', 'xlarge')]
    + [f'vajra-v1-{k}-det.pt' for k in ('nano', 'small', 'medium', 'large', 'xlarge')]
)

GITHUB_ASSETS_DICT = {
    "visdrone": {
        "weights": [f'visdrone-best-vajra-v1-{k}-det.pt' for k in ('nano', 'small', 'medium', 'large', 'xlarge')],
        "version": "v1.0.0"
    },
    "coco": {
        "weights": [f'vajra-v1-{k}-det.pt' for k in ('nano', 'small', "medium", "large", "xlarge")],
        "version": "v1.0.2"
    }
}
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]

def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3, exist_ok=False):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress = threads <= 1,
                ),
                zip(url, repeat(dir))
            )
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)

def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    # Return downloadable file size in bytes
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))


def get_google_drive_file_info(link):
    file_id = link.split("/d/")[1].split("/view")[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                    f"Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v is token
        cd = response.headers.get("content-disposition")
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename

def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```python
        from vajra.utils.downloads import unzip_file

        dir = unzip_file('path/to/file.zip')
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        if len(top_level_dirs) > 1 or (len(files) > 1 and not files[0].endswith("/")):
            # Zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. ../datasets/coco8
        else:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. ../datasets/coco8

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f"WARNING! Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path

def check_disk_space(url="https://ultralytics.com/assets/coco128.zip", path=Path.cwd(), sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    try:
        r = requests.head(url)  # response
        assert r.status_code < 400, f"URL error for {url}: {r.status_code} {r.reason}"  # check response
    except Exception:
        return True  # requests issue, default to True

    # Check file size
    gib = 1 << 30  # bytes per GiB
    data = int(r.headers.get("Content-Length", 0)) / gib  # file size (GB)
    total, used, free = (x / gib for x in shutil.disk_usage(path))  # bytes

    if data * sf < free:
        return True  # sufficient space

    # Insufficient space
    text = (
        f"WARNING! Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
        f"Please free {data * sf - free:.1f} GB additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False

def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    gdrive = url.startswith("https://drive.google.com/")
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".") / (file or url2file(url))  # URL converted to filename
    if "://" not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        desc = f"Downloading {url if gdrive else clean_url(url)} to '{f}'"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        check_disk_space(url, path=f.parent)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = "sS" * (not progress)  # silent
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl return value {r}"
                else:  # urllib download
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(f"Download failure for {url}. Environment is not online.") from e
                elif i >= retry:
                    raise ConnectionError(f"Download failure for {url}. Retry limit reached.") from e
                LOGGER.warning(f"Download failure, retrying {i + 1}/{retry} {url}...")

    if unzip and f.exists() and f.suffix in ("", ".zip", ".tar", ".gz"):
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # unzip to dir if provided else unzip in place
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # unzip
        elif f.suffix in (".tar", ".gz"):
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()  # remove zip
        return unzip_dir
    

def attempt_download_vajra(file, repo='vayuvahan/vajra', release='0.0.1', **kwargs):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    from vajra.utils import LOGGER

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. '0.0.0') and assets (i.e. ['vajra-v1-small-det.pt', 'vajra-v1-medium-det.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v7.0
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5, **kwargs)
            return file

        # GitHub assets
        assets = [f'vajra-v1-{size}{suffix}.pt' for size in ('nano', 'small', 'medium', 'large', 'xlarge') for suffix in ('-det', '-pose', '-cls', '-seg', '-panoptic')]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # backup gdrive mirror
            safe_download(
                file,
                url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/{tag} or {url3}')

    return str(file)


def get_github_assets(repo="ultralytics/assets", version="latest", retry=False):
    if version != "latest":
        version = f"tags/{version}"  # i.e. tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)  # github api
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:
        r = requests.get(url)  # try again
    if r.status_code != 200:
        LOGGER.warning(f"⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]

def attempt_download_asset(file, repo='NamanMakkar/VayuAI', release='v1.0.1', **kwargs):
    from vajra.utils import SETTINGS
    file = str(file)
    file = Path(file.strip().replace("'", ""))

    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)

    else:
        name = Path(urllib.parse.unquote(str(file))).name
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):
            url = str(file).replace(":/", "://")
            file = url2file(name)
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)

        elif name in GITHUB_ASSETS_NAMES:
            if "visdrone" in name:
                release = GITHUB_ASSETS_DICT["visdrone"]["version"]
            else:
                release = GITHUB_ASSETS_DICT["coco"]["version"]
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)
        
        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)
        
        return str(file)

def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.
        files_to_delete (tuple): The files to be deleted.

    Example:
        ```python
        from vajra.utils.downloads import delete_dsstore

        delete_dsstore('path/to/dir')
        ```

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()

def zip_directory(directory, compress=True, exclude=(".DS_Store", "__MACOSX"), progress=True):
    """
    Zips the contents of a directory, excluding files containing strings in the exclude list. The resulting zip file is
    named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping. Default is True.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        (Path): The path to the resulting zip file.

    Example:
        ```python
        from vajra.utils.downloads import zip_directory

        file = zip_directory('path/to/dir')
        ```
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Unzip with progress bar
    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files_to_zip, desc=f"Zipping {directory} to {zip_file}...", unit="file", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file 