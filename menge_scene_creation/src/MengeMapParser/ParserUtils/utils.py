import argparse
import cv2
import numpy as np


def parse_unknown_args(unknown):
    """
    function handles unknown args given by argparse.ArgumentParser class method parse_unknown_args()
    parses positional arguments (args) as well as keyword arguments (kwargs)

    kwargs must be given via commandline in the form:
        1) -key value
        2) --key value
        3) key=value

    :param unknown: second variable returned from parse_unknown_args() method from argparse.ArgumentParser class
    :return:
        unknown_args:   list of positional arguments [arg1, ..., argN]
        unknown_kwargs: dict of keyword arguments {key1:value2, ..., keyN:valueN}
    """
    unknown_kwargs = {}
    unknown_args = []
    key = ''
    for arg in unknown:
        # remove prefix character from tag
        cleaned_arg = arg.lstrip('-')

        # equal sign hints at key-value pair --> split to handle
        split = cleaned_arg.split('=')

        if len(split) == 1:
            # argument not split by equality sign
            if arg.startswith('-'):
                key = split[0]
                # initialize dict entry with true value
                unknown_kwargs[key] = True
            else:
                if not key:
                    unknown_args.append(cleaned_arg)
                elif unknown_kwargs[key] is True:
                    # if no previous argument assigned to key
                    unknown_kwargs[key] = cleaned_arg
                else:
                    # make dict value to list and append
                    unknown_kwargs[key] = list(unknown_kwargs[key])
                    unknown_kwargs[key].append(cleaned_arg)
        elif len(split) == 2:
            # if argument split by equal sign --> key=value
            unknown_kwargs[split[0]] = split[1]
        else:
            # multiple equal signs in argument
            raise argparse.ArgumentError("max one '=' sign can be provided per argument to split into key value pair")

    return unknown_args, unknown_kwargs


def make_img_binary(img):
    # count occurrences of each color value [0...255] in the image
    counts = np.bincount(img.ravel())
    # get color values with highest (background) and second highest (foreground) frequency
    max2 = counts.argsort()[-2:][::-1]

    if max2[0] > max2[1]:
        # background is lighter than foreground
        threshold_type = cv2.THRESH_BINARY_INV
        threshold = max2[1] + 1
    else:
        # foreground is lighter than background
        threshold_type = cv2.THRESH_BINARY
        threshold = max2[1] - 1

    # threshold image to only contain value for background and value for regions
    _, binary = cv2.threshold(img, thresh=threshold, maxval=255, type=threshold_type)

    return binary


def str2bool(s):
    return str(s).lower() in ['true', '1', 't', 'y', 'yes']
