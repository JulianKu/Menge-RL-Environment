#! /usr/bin/env python3

from xml.etree import ElementTree as ElT
import yaml


def xml_indentation(tree: ElT, level: int = 0):
    """
    format xml tree to have proper indentation (modification happens in place)

    :param tree:    xml tree to format
    :param level:   level at which to start (can be left out in general)
    """

    assert isinstance(tree, ElT.Element), "function only works for structures specified as etree.ElementTree.Element"
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


def dict2etree(parent: ElT, dictionary: dict) -> ElT:
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
                subtree = ElT.SubElement(parent, key_str)
                dict2etree(subtree, dictionary[key])
            else:
                parent.set(key, str(val))

    else:
        parent.text = str(dictionary)

    return parent


def read_yaml(file: str) -> dict:
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
