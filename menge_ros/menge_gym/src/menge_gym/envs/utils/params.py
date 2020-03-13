import xml.etree.ElementTree as ElT
from os.path import isfile
import numpy as np


def parseXML(xml_file: str) -> ElT.Element:
    """
    parse an xml file

    :param xml_file:    str, path to xml file
    :return: ElementTree root
    """
    assert isfile(xml_file), "provided argument xml_file does not point to a valid file"
    tree = ElT.parse(xml_file)
    root = tree.getroot()

    return root


def goal2array(goal: ElT.Element) -> np.ndarray:
    """
    transforms a Menge goal passed as a dict into a numpy array (turning it into circular goal regardless of type before)

    :param goal:    dict, specifying goal
    :return: np.array [x, y, r], specifying goal as center x,y and radius r
    """

    goal = goal.attrib

    if goal['type'].lower() == 'point':
        center_x = float(goal['x'])
        center_y = float(goal['y'])
        radius = 0.0
    elif goal['type'].lower() == 'circle':
        center_x = float(goal['x'])
        center_y = float(goal['y'])
        radius = float(goal['radius'])
    elif goal['type'].lower() == 'obb':
        width = float(goal['width'])
        height = float(goal['height'])
        angle_rad = float(goal['angle']) * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        center_x = float(goal['x']) + cos_a * width / 2 - sin_a * height / 2
        center_y = float(goal['y']) + cos_a * height / 2 + sin_a * width / 2
        radius = min(width, height)
    elif goal['type'].lower() == 'aabb':
        min_p = np.array([goal['min_x'], goal['min_y']], dtype=float)
        max_p = np.array([goal['max_x'], goal['max_y']], dtype=float)
        dims = max_p - min_p
        radius = dims.min()
        center_x, center_y = min_p + dims / 2
    else:
        raise ValueError("invalid GoalType")

    return np.array((center_x, center_y, radius))


def get_robot_initial_position(scene_xml: str) -> np.ndarray:
    """

    :param scene_xml:       str, path to scene xml file
    :return:                numpy array containing the robot's initial x, y coordinates
    """
    root = parseXML(scene_xml)
    robot_attributes = root.findall("AgentGroup/ProfileSelector[@name='robot']/../Generator/Agent")[0].attrib
    x = float(robot_attributes['p_x'])
    y = float(robot_attributes['p_y'])
    return np.array((x, y)).reshape(-1, 2)
