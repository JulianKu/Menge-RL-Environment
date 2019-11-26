from xml.etree import ElementTree as ET
import yaml
import numpy as np
from skimage import measure

def xml_indentation(tree, level=0):
    """
    format xml tree to have proper indentation (modification happens in place)

    :param tree:    xml tree to format
    :param level:   level at which to start (can be left out in general)
    """

    assert isinstance(tree, ET.Element), "function only works for structures specified as etree.ElementTree.Element"
    assert isinstance(level, int), "level needs to have integer value"

    indent = "\t"
    i = "\n{}".format(indent * level)
    if len(tree):
        if not tree.text or not tree.text.strip():
            tree.text = "{}".format(i + indent)

        if not tree.tail or not tree.tail.strip():
            tree.tail = i

        for subtree in tree:
            xml_indentation(subtree, level + 1)

        if not subtree.tail or not subtree.tail.strip():
            subtree.tail = i
    else:
        if level and (not tree.tail or not tree.tail.strip()):
            tree.tail = i


def read_yaml(file):
    """
    read yaml file

    :param file:        path to yaml config file
    :return: config:    dictionary containing contents from yaml file
    """

    with open(file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            return {}

    return config


def dict2etree(parent, dictionary):
    """
    turn dictionary into xml etree

    :param parent:          ET.Element under which the xml tree should be build
    :param dictionary:      dict that is to be turned into the xml etree

    :return:                parent, xml tree building happens in-place
    """

    if isinstance(dictionary, dict):
        for key, val in dictionary.items():
            if key.startswith("AgentProfile"):
                keystr = "AgentProfile"
            else:
                keystr = key
            if isinstance(val, dict):
                subtree = ET.SubElement(parent, keystr)
                dict2etree(subtree, dictionary[key])
            else:
                parent.set(key, str(val))

    else:
        parent.text = str(dictionary)

    return parent


def remove_inner_contours(contours):
    """
    removes all contours that lie within other contours

    :param contours:            list of contours specified as numpy arrays (n,2)

    :return: reduced_contours:  contours where inner contours are removed
    """
    inner_contours = set()
    for contour1 in contours:
        for idx, contour2 in enumerate(contours):
            if not np.array_equal(contour1, contour2):
                # get mask that tells which points of contour2 lie within contour1
                pnt_in_cnt2 = measure.points_in_poly(contour2, contour1)
                if np.all(pnt_in_cnt2):
                    # if all points of one contours lie within another, add to set of contours to delete
                    inner_contours.add(idx)

    reduced_contours = np.array(contours)
    return list(np.delete(reduced_contours, list(inner_contours)))


def approximate_contours(contours, tolerance):
    """
    approximates each contour within contours with a polygon given the specified tolerance

    :param contours:        list of contours specified as numpy arrays (n,2)
    :param tolerance:       float, maximum distance from original points of polygon to approximated polygonal chain

    :return:                list of approximated contours
    """
    approx_contours = []
    for contour in contours:
        approx_contours.append(measure.approximate_polygon(contour, tolerance))

    return approx_contours
