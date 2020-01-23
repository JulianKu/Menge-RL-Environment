#! /usr/bin/env python3

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
from skimage import measure, draw
from skimage.morphology import skeletonize
from .ParserUtils.contours_manipulation import remove_inner_contours, approximate_contours
from .ParserUtils.utils import make_img_binary, str2bool
from .ParserUtils.markup_utils import xml_indentation, read_yaml, dict2etree
from .ParserUtils.coordinate_transform import pixel2meter, center2corner_pivot
from .ParserUtils.triangulation import get_triangles, triangulate_map
from .MengeUtils.objToNavMesh import buildNavMesh
from .MengeUtils.ObjReader import ObjFile


class MengeMapParser:
    """
    Take an image (map) and its resolution as input, extract obstacles and build a Menge-compliant
    xml out of that map
    """

    def __init__(self, img_path, resolution, config_path=None, output=None):
        """
        :param img_path:    path to the map image file
        :param resolution:  map resolution in [m]
        :param config_path: path to yaml config file specifying all simulation parameters
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
        self.dims = self.img.shape

        target_path = os.path.join(self.img_dir, self.img_name + "_regions" + self.img_ext)
        if os.path.isfile(target_path):
            self.target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.target = None
        # initialize rect coordinates/mask for each target region box
        self.target_boxes = None
        self.target_idx = None

        trajectory_path = os.path.join(self.img_dir, self.img_name + "_trajectory" + self.img_ext)
        if os.path.isfile(trajectory_path):
            self.trajectory_img = cv2.imread(trajectory_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.trajectory_img = None

        self.resolution = resolution
        if config_path:
            self.config = read_yaml(config_path)
        else:
            self.config = {}

        if output:
            base, ext = os.path.splitext(output)
            if ext:
                # if output contains dir + filename --> take that as output
                assert ext == '.xml', 'Invalid output flag\n Output flag either needs to be a directory ' \
                                      'or an ".xml"-file'
                dirname, filename = os.path.split(output)
                filename = os.path.splitext(filename)[0]
                config_dir = os.path.join(dirname, filename)
                if not os.path.isdir(config_dir):
                    # if dir is not yet existing --> make dir
                    os.makedirs(config_dir)
                self.output = {"base": output,
                               "scene": os.path.join(config_dir, filename + "S" + ext),
                               "behavior": os.path.join(config_dir, filename + "B" + ext),
                               "view": os.path.join(config_dir, filename + "V" + ext),
                               "navmesh": os.path.join(config_dir, filename + "mesh.nav")}
            else:
                config_dir = os.path.join(output, self.img_name)
                if not os.path.isdir(config_dir):
                    os.makedirs(config_dir)
                # if output only contains dir --> infer filename from input image
                self.output = {"base": os.path.join(output, self.img_name + ".xml"),
                               "scene": os.path.join(config_dir, self.img_name + "S.xml"),
                               "behavior": os.path.join(config_dir, self.img_name + "B.xml"),
                               "view": os.path.join(config_dir, self.img_name, self.img_name + "V.xml"),
                               "navmesh": os.path.join(config_dir, self.img_name, self.img_name + "mesh.nav")}
        else:
            # if nothing specified --> infer dir and filename from input image
            config_dir = os.path.join(self.img_dir, self.img_name)
            if not os.path.isdir(config_dir):
                os.makedirs(config_dir)
            self.output = {"base": os.path.join(self.img_dir, self.img_name + ".xml"),
                           "scene": os.path.join(config_dir, self.img_name + "S.xml"),
                           "behavior": os.path.join(config_dir, self.img_name + "B.xml"),
                           "view": os.path.join(config_dir, self.img_name + "V.xml"),
                           "navmesh": os.path.join(config_dir, self.img_name + "mesh.nav")}

        dump_path = os.path.join(config_dir, 'images')
        if not os.path.isdir(dump_path):
            os.makedirs(dump_path)
        self.output["dumpPath"] = dump_path

        # flag whether edge extractor has already been called
        self.obstacles_extracted = False
        # flag whether target area extractor has already been called
        self.targets_extracted = False

        # empty intermediates for contour extraction
        self.thresh = None
        self.closed = None
        self.dilated = None
        self.eroded = None
        self.contours = None
        self.bounds = None
        self.contours_img = None

        # empty image for triangulation
        self.triangle_img = None

        # initialize tolerance for contour approximation
        self.tolerance = None

        # empty xml trees
        self.base_tree = None
        self.scene_tree = None
        self.behavior_tree = None
        self.view_tree = None
        # empty navmesh
        self.navmesh = None

    def full_process(self, **kwargs):
        """
        first extract obstacles from image, then write results into Menge compliant xml file
        """
        self.extract_trajectory()
        self.extract_obstacles(**kwargs)
        self.extract_target_areas()
        self.make_xml(**kwargs)

    def extract_trajectory(self):
        """
        extract trajectory pixels from corresponding image
        if no trajectory given, return image center

        :return: indices of trajectory pixels
        """

        if self.trajectory_img is not None:
            trajectory = make_img_binary(self.trajectory_img) / 255.
            trajectory = skeletonize(trajectory)
            return np.array(np.where(trajectory)).T
        else:
            # return image center
            return np.array(self.img.shape).reshape(-1, 2) // 2

    def extract_obstacles(self, r_d=15, r_c=20, r_e=10, tolerance=0.1, **kwargs):
        """
        extract static obstacles from image by detecting lines/contours

        :param r_d:             radius for structuring element for dilation
        :param r_c:             radius for structuring element for closing
        :param r_e:             radius for structuring element for erosion
        :param tolerance:       tolerance value in [m] for the obstacle approximation
        """

        # make sure arguments are of the correct type (e.g when passed via commandline)
        r_d, r_c, r_e = int(r_d), int(r_c), int(r_e)
        tolerance = float(tolerance)

        ungrayed = self.img.copy()
        # pixel without information got initially assigned gray (128) --> make white (255) instead
        ungrayed[ungrayed == 128] = 255
        # threshold only keep dark pixel --> obstacles
        _, thresh = cv2.threshold(ungrayed, 100, 255, cv2.THRESH_BINARY_INV)
        # closing + dilation to connect structures
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_d, r_d))
        dilated = cv2.dilate(thresh, kernel_dilation)
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_c, r_c))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_closing)
        # erosion to slim out obstacles again
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_e, r_e))
        eroded = cv2.erode(closed, kernel_erosion)

        # get trajectory for traversable area
        trajectory_pixels = self.extract_trajectory()

        # extract contours
        contours = measure.find_contours(eroded, 1, fully_connected='high', positive_orientation='high')
        contours = [np.rint(contour).astype('int32') for contour in contours]
        # remove inner contours
        contours, bounds = remove_inner_contours(contours, trajectory_pixels)
        self.contours = approximate_contours(contours, tolerance / self.resolution)
        self.bounds = approximate_contours(bounds, tolerance / self.resolution)

        # save intermediate results for plotting
        self.thresh = thresh
        self.closed = closed
        self.dilated = dilated
        self.eroded = eroded

        # tolerance required for when target regions are extracted based on obstacles
        self.tolerance = tolerance

        # set flag
        self.obstacles_extracted = True

    def extract_target_areas(self):
        """
        extract rectangular regions from the target image
        """
        if self.target is not None:
            target = make_img_binary(self.target)

            # extract contours
            try:
                # for OpenCV2 and OpenCV4 findContours returns 2 values
                tgt_contours, _ = cv2.findContours(target.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
            except ValueError:
                # for OpenCV3 findContours returns 3 values
                _, tgt_contours, _ = cv2.findContours(target.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for idx, contour in enumerate(tgt_contours):
                tgt_contours[idx] = cv2.minAreaRect(contour)

            tgt_image = np.zeros_like(self.img, dtype='bool')
            for contour in tgt_contours:
                contour = cv2.boxPoints(contour)
                contour = np.rint(contour).astype('int32')
                rr, cc = draw.polygon(contour[:, 1], contour[:, 0])
                tgt_image[rr, cc] = 1

        else:
            # if no target regions specified
            # --> make everything outside the obstacles and inside the bounds a target region
            if not self.obstacles_extracted:
                print("Edges have not been extracted before. Running edge extractor with default settings now")
                self.extract_obstacles()

            # Create a pixel maps of contours
            contour_image = np.zeros_like(self.img, dtype='float')
            bounds_image = np.zeros_like(self.img, dtype='float')
            for contour in self.contours:
                rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
                contour_image[rr, cc] = 1

            smallest_bounding_cnt = self.bounds[0]
            rr, cc = draw.polygon(smallest_bounding_cnt[:, 0], smallest_bounding_cnt[:, 1])
            bounds_image[rr, cc] = 1

            clearance_to_contours = 1  # meter
            kernel_size = (np.rint(clearance_to_contours / self.resolution).astype('int32'),) * 2
            kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            contour_image = cv2.dilate(contour_image, kernel_erosion).astype('bool')
            bounds_image = cv2.erode(bounds_image, kernel_erosion).astype('bool')

            # only allow targets inside of bounding contours where there are no obstacles
            tgt_image = np.bitwise_and(bounds_image, np.bitwise_not(contour_image))

            # maybe sample rectangles in free space? i.e tgt_image == True
            tgt_contours = []

        self.target_boxes = tgt_contours
        self.target_idx = np.where(tgt_image)

        self.targets_extracted = True

    def plot_image(self, image=None):
        """
        plot image via matplotlib.pyplot
        a) if called without arguments --> plot image assigned to instance at instantiation
        b) specify image as np.array (2D or 3D)
        c) specify image by tag (str)

        :param image:   must be either 2D (w,h) or 3D (w,h,c) numpy array or one of then following identifiers (str):
                        "orig"             -- plot input image
                        "thresh"           -- plot input image but only keep dark pixel
                        "dilation"         -- plot morphological dilation of the thresh image
                        "closing"          -- plot morphological closing of the dilated image
                        "erosion"          -- plot morphological erosion of the closed image
                        "contours"         -- plot extracted contours from input image
                        "contours_orig"    -- plot extracted contours over input image
                        "targets"          -- plot extracted targets from target image
                        "targets_orig"     -- plot extracted targets over input image
                        "triangles"        -- plot triangulated map from extracted contours
                        "triangles_orig"   -- plot triangulated map over input image
        """

        if image is None:
            # if nothing specified --> plot original image
            plt.title(' '.join(['map', self.img_name]))
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
                plt.title(' '.join(['map', self.img_name]))
                plt.imshow(self.img, cmap='gray')

            else:
                # if edge extractor has not been called before --> run first to be able to plot
                if not self.obstacles_extracted:
                    print("Edges have not been extracted before. Running edge extractor with default settings now")
                    self.extract_obstacles()

                plt.title(' '.join(['map', self.img_name, '-', image]))

                if image == "thresh":
                    plt.imshow(255 - self.thresh, cmap='gray')
                elif image == "closing":
                    plt.imshow(255 - self.closed, cmap='gray')
                elif image == "dilation":
                    plt.imshow(255 - self.dilated, cmap='gray')
                elif image == "erosion":
                    plt.imshow(255 - self.eroded, cmap='gray')
                elif image == "contours" or image == "contours_orig":
                    if image == "contours":
                        base_image = np.zeros_like(self.img, dtype=np.uint8)
                        alpha = 1
                    else:
                        base_image = self.img.copy()
                        alpha = 0.5

                    plt.imshow(base_image, cmap='gray')

                    base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)

                    for contour in self.contours + self.bounds:
                        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, alpha=alpha)
                        rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1])
                        base_image[rr, cc] = (1 - alpha) * base_image[rr, cc] \
                                             + alpha * np.ones(base_image.shape)[rr, cc] * np.array([255, 0, 0])

                    self.contours_img = base_image

                elif image == "targets" or image == "targets_orig":
                    if not self.targets_extracted:
                        print("Targets have not been extracted before. "
                              "Running target extractor with default settings now")
                        self.extract_target_areas()

                    if image == "targets":
                        base_image = np.zeros_like(self.img, dtype=np.uint8)
                        alpha = 1
                    else:
                        base_image = self.img.copy()
                        alpha = 0.5

                    plt.imshow(base_image, cmap="gray")

                    tgt_image = np.zeros(shape=self.dims + (3,), dtype=np.uint8)
                    tgt_image[self.target_idx] = np.array([0, 255, 0])

                    plt.imshow(tgt_image, alpha=alpha)

                elif image == "triangles" or image == "triangles_orig":

                    if image == "triangles":
                        base_image = np.zeros_like(self.img, dtype=np.uint8)
                        alpha = 1
                    else:
                        base_image = self.img.copy()
                        alpha = 0.5

                    plt.imshow(base_image, cmap='gray')

                    base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)

                    triangles, vertices = get_triangles(self.contours, self.bounds)

                    for triangle in vertices[triangles]:
                        plt.plot(triangle[:, 1], triangle[:, 0], linewidth=1, alpha=alpha)
                        rr, cc = draw.polygon_perimeter(triangle[:, 0], triangle[:, 1])
                        base_image[rr, cc] = (1 - alpha) * base_image[rr, cc] \
                                             + alpha * np.ones(base_image.shape)[rr, cc] * np.array([0, 0, 255])
                    self.triangle_img = base_image

                else:
                    raise ValueError("image must correspond to one of the following identifiers\n"
                                     "{:<15}".format("\"orig\"")
                                     + " -- plot input image\n"
                                       "{:<15}".format("\"thresh\"")
                                     + " -- plot input image but only keep dark pixel\n"
                                       "{:<15}".format("\"dilation\"")
                                     + " -- plot morphological dilation of the thresh image\n"
                                       "{:<15}".format("\"closing\"")
                                     + " -- plot morphological closing of the dilated image\n"
                                       "{:<15}".format("\"erosion\"")
                                     + " -- plot morphological erosion of the closed image\n"
                                       "{:<15}".format("\"contours\"")
                                     + " -- plot extracted contours from input image\n"
                                       "{:<15}".format("\"contours_orig\"")
                                     + " -- plot extracted contours over input image\n"
                                       "{:<15}".format("\"targets\"")
                                     + " -- plot extracted targets from target image\n"
                                       "{:<15}".format("\"targets_orig\"")
                                     + " -- plot extracted targets over input image\n"
                                       "{:<15}".format("\"triangles\"")
                                     + " -- plot triangulated map from extracted contours\n"
                                       "{:<15}".format("\"triangles_orig\"")
                                     + " -- plot triangulated map over input image\n"
                                     )
        else:
            raise ValueError("Either specify image directly via an array (np.ndarray) or via a name tag (str)")

        plt.xticks([])
        plt.yticks([])
        plt.show()

    def make_xml(self, make_navmesh=True, **kwargs):
        """
        compose a Menge compliant scenario out of four xml files (base, scene, behavior, view)
        """

        # make sure arguments are of the correct type (e.g when passed via commandline)
        make_navmesh = str2bool(make_navmesh)

        assert self.config, \
            "Unable to parse config file.\n Config file is required for generating Menge compliant xml files"

        self.make_base(**kwargs)
        self.make_scene(**kwargs)
        self.make_behavior(make_navmesh, **kwargs)
        self.make_view()
        if make_navmesh:
            self.make_navmesh()

    def make_base(self, pedestrian_model="pedvo", **kwargs):
        """
        make a Menge simulator compliant xml file that specifies a scenario based on the scene, behavior and view file.
        """

        assert pedestrian_model in ['pedvo', 'orca'], "Specified pedestrian model is not supported"

        output = self.output

        root = ET.Element("Project")
        for key in output:
            if key != 'base':
                # make attribute for every element in the output dict (scene, behavior, view, dumpPath)
                root.set(key, os.path.relpath(output[key], os.path.split(output['base'])[0]))
        # set pedestrian model for simulation
        root.set("model", pedestrian_model)

        # prettify xml by indentation
        xml_indentation(root)

        self.base_tree = ET.ElementTree(root)

        # write to file
        self.base_tree.write(self.output['base'], xml_declaration=True, encoding='utf-8', method="xml")

    def make_scene(self, num_agents=200, **kwargs):
        """
        make a Menge simulator compliant scene xml file out of the extracted contours and the scene config
        """

        # make sure arguments are of the correct type (e.g when passed via commandline)
        num_agents = int(num_agents)

        assert self.config['Experiment'], \
            "Unable to parse Experiment field in config file.\n " \
            "Config file is required for generating Menge compliant xml files"

        root = ET.Element("Experiment")

        dict2etree(root, self.config['Experiment'])

        # define agent group
        agent_group = ET.SubElement(root, "AgentGroup")
        profile_selector = ET.SubElement(agent_group, "ProfileSelector")
        profile_selector.set("type", "const")
        profile_selector.set("name", "group1")
        state_selector = ET.SubElement(agent_group, "StateSelector")
        state_selector.set("type", "const")
        state_selector.set("name", "Walk")
        generator = ET.SubElement(agent_group, "Generator")
        generator.set("type", "explicit")

        # define agents
        res = self.resolution
        dims = self.dims
        transformed_tgts = pixel2meter(self.target_idx, dims, res)

        for a in range(int(num_agents)):
            agent = ET.SubElement(generator, "Agent")
            random_idx = np.random.choice(len(transformed_tgts[0]))
            agent_x = transformed_tgts[0][random_idx]
            agent_y = transformed_tgts[1][random_idx]
            agent.set("p_x", str(agent_x))
            agent.set("p_y", str(agent_y))

        # define obstacle set
        obstacle_set = ET.SubElement(root, "ObstacleSet")
        obstacle_set.set("type", "explicit")
        obstacle_set.set("class", "1")

        if self.contours is None:
            print("extract_obstacles has not yet been called. ObstacleSet will be empty.")

        # make obstacle for every contour
        for contour in self.contours + self.bounds:
            obstacle = ET.SubElement(obstacle_set, "Obstacle")
            obstacle.set("closed", "1")
            for point in contour:
                obs_point_x, obs_point_y = pixel2meter(point, dims, res)
                vertex = ET.SubElement(obstacle, "Vertex")
                vertex.set("p_x", str(obs_point_x))
                vertex.set("p_y", str(obs_point_y))

        # prettify xml by indentation
        xml_indentation(root)

        self.scene_tree = ET.ElementTree(root)

        # write to file
        self.scene_tree.write(self.output['scene'], xml_declaration=True, encoding='utf-8', method="xml")

    def make_behavior(self, make_navmesh=False, num_goals=None, **kwargs):
        """
        make a Menge simulator compliant behavior xml file out of the behavior config

        :param: num_goals: only required when no target is given,
                           number of goals to sample from free space in environment (self.target_idx)
        """

        # make sure arguments are of the correct type (e.g when passed via commandline)
        if num_goals:
            num_goals = int(num_goals)

        assert self.config['BFSM'], \
            "Unable to parse BFSM behavior field in config file.\n " \
            "Config file is required for generating Menge compliant xml files"

        root = ET.Element("BFSM")

        goalset = ET.SubElement(root, "GoalSet")
        goalset.set("id", str(0))

        res = self.resolution
        dims = self.dims
        if self.target_boxes:
            # create goal areas from boxes
            tgt_boxes = self.target_boxes
            for tgt_id, tgt_box in enumerate(tgt_boxes):
                goal = ET.SubElement(goalset, "Goal")
                goal.set("id", str(tgt_id))
                goal.set("capacity", str(1000))
                goal.set("type", "OBB")
                # tgt_box is defined as (c, r) of center, (height, width), angle
                # first: transform pixel values to metric scale
                # --> change order of center coordinates to be (r, c) instead
                box_x, box_y = pixel2meter(tgt_box[0][::-1], dims, res)
                # map width and height to metric scale as well
                height, width = tuple(map(lambda i: i * res, tgt_box[1]))
                # map angle to turned coordinate system
                angle = 90. - tgt_box[2]
                tgt_box = (box_x, box_y), (width, height), angle
                # perform transformation from pivot = center to pivot = corner
                tgt_box = center2corner_pivot(tgt_box)

                goal.set("x", str(tgt_box[0][0]))
                goal.set("y", str(tgt_box[0][1]))
                goal.set("width", str(tgt_box[1][0]))
                goal.set("height", str(tgt_box[1][1]))
                goal.set("angle", str(tgt_box[2]))
                goal.set("weight", str(1.0))

        else:
            # sample goal points in free space
            assert num_goals, "If no goal regions are specified, you have to at least provide the number of goals"

            transformed_tgts = pixel2meter(self.target_idx, dims, res)
            for tgt_id in range(int(num_goals)):
                random_idx = np.random.choice(len(transformed_tgts[0]))
                tgt_x = transformed_tgts[0][random_idx]
                tgt_y = transformed_tgts[1][random_idx]
                goal = ET.SubElement(goalset, "Goal")
                goal.set("id", str(tgt_id))
                goal.set("capacity", str(1000))
                goal.set("type", "circle")
                goal.set("x", str(tgt_x))
                goal.set("y", str(tgt_y))
                goal.set("radius", str(0.5))
                goal.set("weight", str(1.0))

        if make_navmesh:
            self.config['BFSM']['StateWalk']['VelComponent'] = {'type': 'nav_mesh',
                                                                'file_name': os.path.split(self.output['navmesh'])[1],
                                                                'heading_threshold': '5'}

        dict2etree(root, self.config['BFSM'])

        # prettify xml by indentation
        xml_indentation(root)

        self.behavior_tree = ET.ElementTree(root)

        # write to file
        self.behavior_tree.write(self.output['behavior'], xml_declaration=True, encoding='utf-8', method="xml")

    def make_view(self):
        """
        make a Menge simulator compliant view xml file out of the view config
        """

        assert self.config, \
            "Unable to parse config file.\n Config file is required for generating Menge compliant xml files"

        #TODO:
        # find way to specify view automatically from map
        pass

    def make_navmesh(self):

        # triangulate map --> extract vertices and faces
        verts, faces = triangulate_map(self.contours, self.bounds)
        verts = np.transpose(pixel2meter(np.transpose(verts), self.dims, self.resolution))
        print("\t {:d} faces have been extracted from the map".format(len(faces)))

        # write Wavefront obj file from map
        obj_file = os.path.splitext(self.output['navmesh'])[0] + '.obj'
        with open(obj_file, 'w') as write_file:
            write_file.write("# OBJ file for %s\n" % self.img_name)
            write_file.write("\n")
            write_file.write("# Vertices\n")
            for vert in verts:
                write_file.write("v {0:.4f} {1:.4f} {2:.4f}\n".format(vert[0], vert[1], 0))
            write_file.write("\n")
            write_file.write("# Normal\n")
            write_file.write("vn {0:.4f} {1:.4f} {2:.4f}\n".format(0, 0, 1))
            write_file.write("\n")
            write_file.write("# Faces\n")
            for face in faces:
                write_file.write("f {0:d}//1 {1:d}//1 {2:d}//1\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

        obj = ObjFile(obj_file)
        gCount, fCount = obj.faceStats()
        print("\tObj File has {:d} faces".format(fCount))

        mesh = buildNavMesh(obj, y_up=False)

        mesh.writeNavFile(self.output['navmesh'], ascii=True)

        self.navmesh = mesh
