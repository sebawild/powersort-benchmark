## Powersort in Numpy

This code snipped was used during development of the [final implementation](https://github.com/numpy/numpy/pull/29208) of Powersort in NumPy.

As the real codebase in the NumPy repository has dependencies on multiple other files, the needed parts are copied into a brief "setup" section at the start of the file.

In addition to "classical" sorting (i.e. the original elements are returned in ascending/descending order), NumPy also provides support for [argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html),
in which the permutation that sorts the array when applied gets returned. Following the convention from numpy, argsort-related functions are prefixed with "a".