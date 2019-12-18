from xml.etree import ElementTree as ET
import yaml
import numpy as np
from skimage import measure
import cv2



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
                key_str = "AgentProfile"
            elif key.startswith("State"):
                key_str = "State"
            elif key.startswith("Transition"):
                key_str = "Transition"
            else:
                key_str = key
            if isinstance(val, dict):
                subtree = ET.SubElement(parent, key_str)
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

    def cnt_length(cnt):
        diff = cnt - np.roll(cnt, 1, axis=0)
        return np.sum(np.sqrt(np.sum(diff**2, axis=1)))

    approx_contours = []
    for contour in contours:
        # make sure contour is closed
        if not np.array_equiv(contour[0], contour[-1]):
            contour = np.append(contour, contour[0]).reshape(-1, 2)
        # only polygons of higher order than triangles need to be approximated
        if len(contour) > 3:
            approx = measure.approximate_polygon(contour, tolerance)
            # remove obstacles smaller than the tolerance
            if cnt_length(approx) > tolerance:
                approx_contours.append(approx)
        # remove obstacles smaller than the tolerance
        elif cnt_length(contour) > tolerance:
            approx_contours.append(contour)

    return approx_contours


def pixel2meter(coordinates, dimension, resolution):
    """
    transforms the pixel coordinate of a map image to the corresponding metric dimensions
        NOTE: pixel coordinates start in the upper left corner with (0,0)
              metric coordinates start in the lower left corner with (0,0)

    :param coordinates: pixel coordinates (r, c)
    :param dimension: dimensions of the image in total
    :param resolution: resolution of the map [m] ( 1 pixel = ? meter )

    :return: coordinates: metric coordinates (x, y)
    """

    x = coordinates[1] * resolution  # column * resolution
    y = (dimension[0] - coordinates[0]) * resolution  # ( dim_r - r ) * resolution

    return x, y


def center2corner_pivot(box):
    """
    transformes a box, defined by x and y of the center, width, height and rotation angle (rotation around center)
    into a box, defined by x and y of the lower corner, width, height and rotation angle (rotation around lower corner)

    :param box: tuple of center (tuple x, y), size (tuple width, height) and angle

    :return: box: tuple of pivot lower left corner (tuple x, y), size (tuple width, height) and angle
    """

    center, size, angle = box
    center_x, center_y = center
    width, height = size
    cos_a = np.cos(angle * np.pi / 180)
    sin_a = np.sin(angle * np.pi / 180)
    pivot_x = center_x - cos_a * width/2 + sin_a * height/2
    pivot_y = center_y - cos_a * height/2 - sin_a * width/2

    return (pivot_x, pivot_y), (width, height), angle

