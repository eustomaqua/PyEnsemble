# coding: utf8
# Aim to: read data for experiments, (codes about images are from keras)
#         data_keras.py (original)
# Available for:



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()

import os
import sys
import time

import numpy as np 




#======================================
# /usr/local/lib/python2.7/dist-packages/keras/backend
#======================================



#--------------------------------------
# __init__.py
#--------------------------------------


# from .common import image_data_format



#--------------------------------------
# common.py
#--------------------------------------


_FLOATX = 'float32'
_EPSILON = 10e-8
_IMAGE_DATA_FORMAT = 'channels_last'


def image_data_format():
    """Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
    """Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    """
    global _IMAGE_DATA_FORMAT
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown data_format:', data_format)
    _IMAGE_DATA_FORMAT = str(data_format)



#--------------------------------------
#
#--------------------------------------




#======================================
# /usr/local/lib/python2.7/dist-packages/keras/preprocessing
#======================================



#--------------------------------------
# sequence.py
#--------------------------------------


# from __future__ import absolute_import
#
# import numpy as np 
# import random
# from six.moves import range


def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: int, maximum length
        seq: list of lists where each sublist is a sequence
        label: list where each element is an integer

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label



#--------------------------------------
# 
#--------------------------------------



#======================================
# /usr/local/lib/python2.7/dist-packages/keras/utils
#======================================



#--------------------------------------
#
#--------------------------------------


import time
import sys
import six
import marshal
import types as python_types
import inspect

_GLOBAL_CUSTOM_OBJECTS = {}

import hashlib
import multiprocessing
import os
import random
import shutil
# import sys
import tarfile
import threading
# import time
import zipfile
from abc import abstractmethod
from multiprocessing.pool import ThreadPool
#
# import numpy as np
# import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
# 
try:
    import queue
except ImportError:
    import Queue as queue
# from ..utils.generic_utils import Progbar



#--------------------------------------
#
#--------------------------------------


if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.

        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.

        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """

        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve



#--------------------------------------
#
#--------------------------------------


def _extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    # Returns
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format is 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type is 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type is 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


# def get_file():



#--------------------------------------
# from ..utils.generic_utils import Progbar
#--------------------------------------


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self.start)
        if self.verbose == 1:
            if (not force and (now - self.last_update) < self.interval and
                    current < self.target):
                return

            prev_total_width = self.total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self.total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (' ' * (prev_total_width - self.total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)



#--------------------------------------
# from ..utils.data_utils import get_file
#--------------------------------------


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    # Example

    ```python
        >>> from keras.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        The file hash
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    # Returns
        Whether the file is valid
    """
    if ((algorithm is 'sha256') or
            (algorithm is 'auto' and len(file_hash) is 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).

    # Returns
        Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' +
                      file_hash + ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size is -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath



#--------------------------------------
#
#--------------------------------------



#======================================
# /usr/local/lib/python2.7/dist-packages/keras/datasets
#======================================



#--------------------------------------
# mnist.py
#--------------------------------------


def mnist__load_data(path='mnist.npz'):
    """ Loads the MNIST dataset. 

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path, 
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz', 
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)



#--------------------------------------
# cifar.py
#--------------------------------------


from six.moves import cPickle


def load_batch(fpath, label_key='labels'):
    ## from six.moves import cPickle
    """ Internal utility for parsing CIFAR data. 

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve 
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels



#--------------------------------------
# cifar10.py
#--------------------------------------


# from __future__ import absolute_import
#
# from .cifar import load_batch
# from ..utils.data_utils import get_file
# from .. import backend as K
# import numpy as np 
# import os


def cifar10__load_data():
    """ Loads CIFAR10 dataset. 

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :], 
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # if K.image_data_format() == 'channels_last':
    if image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)



#--------------------------------------
# cifar100.py
#--------------------------------------


# from __future__ import absolute_import
# 
# from .cifar import load_batch
# from ..utils.data_utils import get_file
# from .. import backend as K 
# import numpy as np
# import os


def cifar100__load_data(label_mode='fine'):
    """ Loads CIFAR100 dataset. 

    # Arguments
        label_mode: one of "fine", "coarse".

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # if K.image_data_format() == 'channels_last':
    if image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)



#--------------------------------------
# fashion_mnist.py
#--------------------------------------


import gzip
# import os
# 
# from ..utils.data_utils import get_file
# import numpy as np


def fashion_mnist__load_data():
    """ Loads the Fashion-MNIST dataset. 

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for file in files:
        paths.append(get_file(file, origin=base + file, cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, 
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, 
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)



#--------------------------------------
# reuters.py
#--------------------------------------


# from __future__ import absolute_import
#
# from ..utils.data_utils import get_file
# from ..preprocessing.sequence import _remove_long_seq
# from six.moves import zip
# import numpy as np
import json
import warnings


def reuters__load_data(path='reuters.npz', num_words=None, skip_top=0, 
                       maxlen=None, test_split=0.2, seed=113, 
                       start_char=1, oov_char=2, index_from=3, **kwargs):
    """ Loads the Reuters newswire classification dataset. 

    # Arguments
        path: where to cache the data (relative to `~/.keras/datasets`).
        num_words: max number of words to include. Words are ranked 
            by how often they occur (in the training set) and only 
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative). 
        maxlen: truncate sequences after this length. 
        test_split: Fraction of the dataset to be used as test data. 
        seed: random seed for sample shuffling. 
        start_char: The start of a sequence will be marked with this character. 
            Set to 1 because 0 is usually the padding character. 
        oov_char: words that were cut out because of the `num_words` 
            or `skip_top` limit will be replaced with this character. 
        index_from: index actual words with this index and higher. 

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Note that the 'out of vocabulary' character is only used for 
    words that were present in the training set but are not included 
    because they're not making the `num_words` cut here. 
    Words that were not seen in the training set but are in the test set 
    have simply been skipped. 
    """
    # legacy support
    if 'nb_words' in kwargs:
        warnings.warn('The `nb_words` argument in `load_data` '
                      'has been renamed `num_words`.')
        num_words = kwargs.pop('nb_words')
    if kwargs:
        raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

    path = get_file(path, 
                    origin='https://s3.amazonaws.com/text-datasets/reuters.npz', 
                    file_hash='87aedbeb0cb229e378797a632c1997b6')
    with np.load(path) as f:
        xs, labels = f['x'], f['y']

    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)

    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if (skip_top <= w < num_words)] for x in xs]

    idx = int(len(xs) * (1 - test_split))
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)



#--------------------------------------
#
#--------------------------------------



#======================================
# 
#======================================



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------




#======================================
# 
#======================================



#--------------------------------------
#
#--------------------------------------


#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------




#======================================
# Validation Codes
#======================================



#--------------------------------------
#
#--------------------------------------


'''

print(image_data_format())
set_image_data_format('channels_first')
print(image_data_format())


(x_train, y_train), (x_test, y_test) = mnist__load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))
(x_train, y_train), (x_test, y_test) = cifar10__load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))
(x_train, y_train), (x_test, y_test) = cifar100__load_data(label_mode='coarse')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))
(x_train, y_train), (x_test, y_test) = cifar100__load_data(label_mode='fine')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))
(x_train, y_train), (x_test, y_test) = fashion_mnist__load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))
(x_train, y_train), (x_test, y_test) = reuters__load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, np.unique(np.concatenate([y_train, y_test])))

'''



#--------------------------------------
#
#--------------------------------------


'''
channels_last
channels_first
(60000, 28, 28)    (60000,)   (10000, 28, 28)    (10000,)   [0 1 2 3 4 5 6 7 8 9]
(50000, 3, 32, 32) (50000, 1) (10000, 3, 32, 32) (10000, 1) [0 1 2 3 4 5 6 7 8 9]
(50000, 3, 32, 32) (50000, 1) (10000, 3, 32, 32) (10000, 1) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
(50000, 3, 32, 32) (50000, 1) (10000, 3, 32, 32) (10000, 1) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
(60000, 28, 28) (60000,) (10000, 28, 28) (10000,) [0 1 2 3 4 5 6 7 8 9]
(8982,) (8982,) (2246,) (2246,) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
[Finished in 7.0s]
'''



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------



#--------------------------------------
#
#--------------------------------------


