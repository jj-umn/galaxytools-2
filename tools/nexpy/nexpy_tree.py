#!/usr/bin/env python

import argparse

from nexusformat.nexus import nxload


def __main__():

    parser = argparse.ArgumentParser(
        description='print the tree structure of a NeXus file')
    parser.add_argument(
        'input',
        type=argparse.FileType('rb'),
        help='NeXus File')
    parser.add_argument(
        'output',
        type=argparse.FileType('w'),
        help='NeXus description')
    args = parser.parse_args()

    a = nxload(args.input.name)
    args.output.write(a.tree)


if __name__ == "__main__":
    __main__()
