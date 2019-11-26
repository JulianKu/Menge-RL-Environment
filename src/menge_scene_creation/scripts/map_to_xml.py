import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import rospy as rp
import os
from sys import exit
import matplotlib.pyplot as plt
from skimage import measure
from utils import *


class MapParser:
    """
    Take an image (map) and its resolution as input, extract obstacles and build a Menge-compliant
    xml out of that map
    """

    def __init__(self, img_path, resolution, config_path=None, output=None):
        """
        :param img_path:    path to the map image file
        :param resolution:  map resolution in [m]
        :param output:      name/path of output file; if not given,
                            name + dir will be inferred from the map file instead
        """
        self.full_path = os.path.abspath(img_path)
        assert os.path.isfile(self.full_path), \
            "Unable to parse map file.\n Map_file argument needs to point to a valid file"

        # split up directory, file, image name and extension
        self.img_dir, self.img_file = os.path.split(self.full_path)
        self.img_name, self.img_ext = os.path.splitext(self.img_file)
        # read image
        self.img = cv2.imread(self.full_path, cv2.IMREAD_GRAYSCALE)
        assert self.img is not None, \
            "Unable to parse map file.\n Map_file argument needs to point to an image"

        self.resolution = resolution
        if config_path:
            self.config = read_yaml(config_path)
        else:
            self.config = {}

        if output:
            ext = os.path.splitext(output)[1]
            if ext:
                # if output contains dir + filename --> take that as output
                assert ext == '.xml', 'Invalid output flag\n Output flag either needs to be a directory ' \
                                      'or an ".xml"-file'
                dirname, filename = os.path.split(output)
                if not os.path.isdir(dirname):
                    # if dir is not yet existing --> make dir
                    os.makedirs(dirname)
                self.output = output
            else:
                # if output only contains dir --> infer filename from input image
                self.output = os.path.join(output, self.img_name + "_scene.xml")
        else:
            # if nothing specified --> infer dir and filename from input image
            self.output = os.path.join(self.img_dir, self.img_name + "_scene.xml")

        # flag whether edge extractor has already been called
        self.edges_extracted = False

        # empty intermediates for line extraction
        self.thresh = None
        self.closing = None
        self.dilated = None
        self.lines = None
        self.contours = None

        # empty xml tree
        self.tree = None

    def make_xml(self):
        """
        make an Menge simulator compliant xml file out of the extracted lines
        """
        root = ET.Element("Experiment")
        # root.set('version', '2.0')

        dict2etree(root, self.config['Experiment'])

        agent_group = ET.SubElement(root, "AgentGroup")
        profile_selector = ET.SubElement(agent_group, "ProfileSelector")
        profile_selector.set("t0ype", "const")
        profile_selector.set("name", "group1")
        state_selector = ET.SubElement(agent_group, "StateSelector")
        state_selector.set("type", "const")
        state_selector.set("name", "Walk")
        generator = ET.SubElement(agent_group, "Generator")
        generator.set("type", "explicit")

        # define agents
        num_agents = 20
        res = self.resolution
        dim_x, dim_y = self.img.shape
        for a in range(num_agents):
            agent = ET.SubElement(generator, "Agent")
            p_x = a * res * 0.9 * dim_x / num_agents + 0.1 * dim_x * res
            p_y = (num_agents - a) * res * 0.9 * dim_y / num_agents + 0.1 * dim_y * res
            agent.set("p_x", str(p_x))
            agent.set("p_y", str(p_y))

        obstacle_set = ET.SubElement(root, "ObstacleSet")
        obstacle_set.set("type", "explicit")
        obstacle_set.set("class", "1")

        # make obstacle for every contour
        for contour in self.contours:
            obstacle = ET.SubElement(obstacle_set, "Obstacle")
            obstacle.set("closed", "1")
            for point in contour:
                p_x = point[0]
                p_y = point[1]
                vertex = ET.SubElement(obstacle, "Vertex")
                vertex.set("p_x", str(p_x * res))
                vertex.set("p_y", str(p_y * res))

        # prettify xml by indentation
        xml_indentation(root)

        self.tree = ET.ElementTree(root)

        # write to file
        self.tree.write(self.output, xml_declaration=True, encoding='utf-8', method="xml")

    def extract_obstacles(self, r_d=15, r_c=20, tolerance=0.5):
        """
        extract static obstacles from image by detecting lines/contours

        :param r_d:             radius for structuring element for dilation
        :param r_c:             radius for structuring element for closing
        :param tolerance:       tolerance value in [m] for the obstacle approximation
        """
        ungrayed = self.img.copy()
        # pixel without information got initially assigned gray (128) --> make white (255) instead
        ungrayed[ungrayed == 128] = 255
        # threshold only keep dark pixel --> obstacles
        _, self.thresh = cv2.threshold(ungrayed, 100, 255, cv2.THRESH_BINARY_INV)
        # closing + dilation to connect structures
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_d, r_d))
        self.dilated = cv2.dilate(self.thresh, kernel_dilation)
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_c, r_c))
        self.closing = cv2.morphologyEx(self.dilated, cv2.MORPH_CLOSE, kernel_closing)

        # extract lines
        self.lines = cv2.HoughLinesP(self.closing, 1, np.pi / 180, 40, minLineLength=30, maxLineGap=30)

        # extract contours
        contours = measure.find_contours(self.closing, 1, fully_connected='high', positive_orientation='high')
        contours = [np.rint(contour).astype('int32') for contour in contours]
        # remove inner contours
        contours = remove_inner_contours(contours)
        self.contours = approximate_contours(contours, tolerance / self.resolution)

        self.edges_extracted = True  # set flag

    def full_process(self, **kwargs):
        """
        first extract obstacles from image, then write results into Menge compliant xml file
        """
        self.extract_obstacles(**kwargs)
        self.make_xml()

    def plot_image(self, image=None):
        """
        plot image via matplotlib.pyplot
        a) if called without arguments --> plot image assigned to instance at instantiation
        b) specify image as np.array (2D or 3D)
        c) specify image by tag (str)

        :param image:   must be either 2D (w,h) or 3D (w,h,c) numpy array or one of then following identifiers (str):
                        "orig"             -- plot input image
                        "thresh"           -- plot input image but only keep dark pixel
                        "closing"          -- plot morphological closing of the thresh image\n"
                        "dilclos"          -- plot morphological dilation of the closing image\n"
                        "lines"            -- plot extracted lines from input image\n"
                        "lines_orig"       -- plot extracted lines over input image\n"
                        "contours"         -- plot extracted contours from input image\n"
                        "contours_orig"    -- plot extracted contours over input image"
        """
        if image is None:
            # if nothing specified --> plot original image
            plt.imshow(self.img, cmap="gray")
        elif type(image) == np.ndarray:
            # plot image that is passed as numpy array
            if image.ndim == 2:
                # grayscale image
                plt.imshow(image, cmap="gray")
            elif image.ndim == 3:
                # RGB image
                plt.imshow(image)
            else:
                raise ValueError("Invalid image: contains less than 2 or more than 3 dimensions")
        elif type(image) == str:
            # plot image that is specified via name tag
            if image == "orig":
                plt.imshow(self.img, cmap='gray')
            else:
                # if edge extractor has not been called before --> run first to be able to plot
                if not self.edges_extracted:
                    print("Edges have not been extracted before. Running edge extractor with default settings now")
                    self.extract_obstacles()

                if image == "thresh":
                    plt.imshow(255 - self.thresh, cmap='gray')
                elif image == "closing":
                    plt.imshow(255 - self.closing, cmap='gray')
                elif image == "dilation":
                    plt.imshow(255 - self.dilated, cmap='gray')
                elif image == "lines" or image == "lines_orig":
                    plot = np.zeros(shape=self.img.shape + (3,), dtype=np.uint8)
                    for line in self.lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(plot, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    if image == "lines":
                        plt.imshow(plot)
                    else:
                        plot = cv2.addWeighted(cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2RGB), 0.5, plot, 1, 0)
                        plt.imshow(plot)
                elif image == "contours" or image == "contours_orig":
                    if image == "contours":
                        base_image = np.zeros(shape=self.img.shape, dtype=np.uint8)
                        alpha = 1
                    else:
                        base_image = self.img.copy()
                        alpha = 0.5

                    fig, ax = plt.subplots()
                    ax.imshow(base_image, cmap=plt.cm.gray)

                    for n, contour in enumerate(self.contours):
                        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, alpha=alpha)

                    ax.axis('image')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.show()

                else:
                    raise ValueError("image must correspond to one of the following identifiers\n"
                                     "{:<15}".format("\"orig\"")
                                     + " -- plot input image\n"
                                       "{:<15}".format("\"thresh\"")
                                     + " -- plot input image but only keep dark pixel\n"
                                       "{:<15}".format("\"closing\"")
                                     + " -- plot morphological closing of the thresh image\n"
                                       "{:<15}".format("\"dilclos\"")
                                     + " -- plot morphological dilation of the closing image\n"
                                       "{:<15}".format("\"lines\"")
                                     + " -- plot extracted lines from input image\n"
                                       "{:<15}".format("\"lines_orig\"")
                                     + " -- plot extracted lines over input image\n"
                                       "{:<15}".format("\"contours\"")
                                     + " -- plot extracted contours from input image\n"
                                       "{:<15}".format("\"contours_orig\"")
                                     + " -- plot extracted contours over input image")
        else:
            raise ValueError("Either specify image directly via an array (np.ndarray) or via a name tag (str)")
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Take an image (map) and its resolution as input, '
                                     'extract obstacles and build a Menge-compliant xml out of that map')
    parser.add_argument('map_file', help='path to the map image file which shall be used in Menge')
    parser.add_argument('resolution', type=float, help='map resolution in [m]')
    parser.add_argument('--output_file', '-o', help="name/path of output file; if not given, "
                                                    "name + dir will be inferred from the map file instead")
    args = parser.parse_args(rp.myargv()[1:])

    try:
        if args.o:
            img_parser = MapParser(args.map_file, args.resolution, args.o)
        else:
            img_parser = MapParser(args.map_file, args.resolution)

        img_parser.full_process()

    except (AssertionError, ValueError) as e:
        print("ERROR")
        exit(e)
