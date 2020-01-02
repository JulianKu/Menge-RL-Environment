import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import rospy as rp
import os
from sys import exit
import matplotlib.pyplot as plt
from skimage import measure, draw
from utils import xml_indentation, read_yaml, dict2etree, remove_inner_contours, \
    approximate_contours, pixel2meter, center2corner_pivot, get_triangles, triangulate_map
from MengeUtils.navMesh import Node, Edge, Obstacle, NavMesh
from MengeUtils.primitives import Vector2, Face


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
        self.dims = self.img.shape

        target_path = os.path.join(self.img_dir, self.img_name + "_regions" + self.img_ext)
        if os.path.isfile(target_path):
            self.target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.target = None
        # initialize rect coordinates/mask for each target region box
        self.target_boxes = None
        self.target_idx = None

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

    def full_process(self, **kwargs):
        """
        first extract obstacles from image, then write results into Menge compliant xml file
        """
        self.extract_obstacles(**kwargs)
        self.extract_target_areas()
        self.make_xml(**kwargs)

    def extract_obstacles(self, r_d=15, r_c=20, r_e=10, tolerance=0.5):
        """
        extract static obstacles from image by detecting lines/contours

        :param r_d:             radius for structuring element for dilation
        :param r_c:             radius for structuring element for closing
        :param r_e:             radius for structuring element for erosion
        :param tolerance:       tolerance value in [m] for the obstacle approximation
        """
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

        # extract contours
        contours = measure.find_contours(eroded, 1, fully_connected='high', positive_orientation='high')
        contours = [np.rint(contour).astype('int32') for contour in contours]
        # remove inner contours
        contours = remove_inner_contours(contours)
        self.contours = approximate_contours(contours, tolerance / self.resolution)

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
            # count occurrences of each color value [0...255] in the target image
            counts = np.bincount(self.target.ravel())
            # get color values with highest (background) and second highest (marked regions) frequency
            max2 = counts.argsort()[-2:][::-1]

            if max2[0] > max2[1]:
                # background is lighter than target regions
                threshold_type = cv2.THRESH_BINARY_INV
                threshold = max2[1] + 1
            else:
                # target region is lighter than background
                threshold_type = cv2.THRESH_BINARY
                threshold = max2[1] - 1

            # threshold image to only contain value for background and value for regions
            _, target = cv2.threshold(self.target, thresh=threshold, maxval=255, type=threshold_type)
            # extract contours
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
            # if no target regions specified --> make everything outside the obstacles a target region
            if not self.obstacles_extracted:
                print("Edges have not been extracted before. Running edge extractor with default settings now")
                self.extract_obstacles()

            # Create a contour image
            contour_image = np.zeros_like(self.img, dtype='float')
            for contour in self.contours:
                rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
                contour_image[rr, cc] = 1

            clearance_to_contours = 1  # meter
            kernel_size = (np.rint(clearance_to_contours / self.resolution).astype('int32'),) * 2
            kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            contour_image = cv2.dilate(contour_image, kernel_erosion).astype('bool')

            tgt_image = np.bitwise_not(contour_image)

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

                    for contour in self.contours:
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

                    triangles, vertices = get_triangles(self.contours)

                    for triangle in vertices[triangles]:
                        plt.plot(triangle[:, 1], triangle[:, 0], linewidth=1, alpha=alpha)
                        rr, cc = draw.polygon_perimeter(triangle[:, 0], triangle[:, 1])
                        base_image[rr, cc] = (1 - alpha) * base_image[rr, cc] \
                                             + alpha * np.ones(base_image.shape)[rr, cc] * np.array([255, 0, 0])
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

    def make_xml(self, **kwargs):
        """
        compose a Menge compliant scenario out of four xml files (base, scene, behavior, view)
        """

        assert self.config, \
            "Unable to parse config file.\n Config file is required for generating Menge compliant xml files"

        self.make_base(**kwargs)
        self.make_scene(**kwargs)
        self.make_behavior(**kwargs)
        self.make_view()

    def make_base(self, pedestrian_model="orca"):
        """
        make a Menge simulator compliant xml file that specifies a scenario based on the scene, behavior and view file.
        """

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

    def make_scene(self, num_agents=200):
        """
        make a Menge simulator compliant scene xml file out of the extracted contours and the scene config
        """

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

        for a in range(num_agents):
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
        for contour in self.contours:
            if not len(contour) <= 2:
                if not len(contour) == 3 and cv2.arcLength(contour, True) > 0.3 / 0.05:
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

    def make_behavior(self, num_goals=None):
        """
        make a Menge simulator compliant behavior xml file out of the behavior config

        :param: num_goals: only required when no target is given,
                           number of goals to sample from free space in environment (self.target_idx)
        """

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

                goal.set("edge_idx", str(tgt_box[0][0]))
                goal.set("y", str(tgt_box[0][1]))
                goal.set("width", str(tgt_box[1][0]))
                goal.set("height", str(tgt_box[1][1]))
                goal.set("angle", str(tgt_box[2]))
                goal.set("weight", str(1.0))

        else:
            # sample goal points in free space

            transformed_tgts = pixel2meter(self.target_idx, dims, res)
            for tgt_id in range(num_goals):
                random_idx = np.random.choice(len(transformed_tgts[0]))
                tgt_x = transformed_tgts[0][random_idx]
                tgt_y = transformed_tgts[1][random_idx]
                goal = ET.SubElement(goalset, "Goal")
                goal.set("id", str(tgt_id))
                goal.set("capacity", str(1000))
                goal.set("type", "circle")
                goal.set("edge_idx", str(tgt_x))
                goal.set("y", str(tgt_y))
                goal.set("radius", str(0.5))
                goal.set("weight", str(1.0))

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

        pass

    def make_navmesh(self):

        def processObstacles(obstacles, vertObstMap, vertNodeMap, navMesh):
            """
            Given a list of Obstacle instances, connects the obstacles into sequences such that each obstacle
            points to the appropriate "next" obstacle.  Assigns obstacles to nodes based on vertex.
            Finally, sets the obstacles to the navigation mesh.
            """

            # I'm assuming that the external edges form perfect, closed loops
            #   That means if a vertex is incident to an obstacle, then it must be incident to two and only
            #   two obstacles.  This tests that assumption
            degrees = map(lambda x: len(x), vertObstMap.values())
            assert (sum(map(lambda x: x % 2, degrees)) == 0)

            # now connect them up
            #   - this assumes that they are all wound properly
            for vertID in vertObstMap.keys():
                o0, o1 = vertObstMap[vertID]
                obst0 = obstacles[o0]
                obst1 = obstacles[o1]
                if obst0.v0 == vertID:
                    obst1.next = o0
                else:
                    obst0.next = o1
                # The obstacle should be in the set of every node built on this vertex
                for node in vertNodeMap[vertID]:
                    node.addObstacle(o0)
                    node.addObstacle(o1)

            # all obstacles now have a "next" obstacle
            assert (len(list(filter(lambda x: x.next == -1, obstacles))) == 0)

            navMesh.obstacles = obstacles
            # end of function

        verts, edges, faces, elev = triangulate_map(self.dims, self.contours, 5 / self.resolution)
        verts = np.transpose(pixel2meter(np.transpose(verts), self.dims, self.resolution))
        navMesh = NavMesh()
        navMesh.vertices = list(map(tuple, verts))
        vertNodeMap = {}
        edgeMap = {}
        nodes = []

        for f, face in enumerate(faces):
            face_verts = face['verts']
            face_edges = face['edges']
            num_verts = len(face_verts)
            node = Node()
            faceObj = Face(v=list(face_verts))
            node.poly = faceObj
            A = B = C = 0.0
            M = []
            b = []
            center_2d = Vector2(0, 0)

            for vert_idx in face_verts:
                if not vert_idx in vertNodeMap:
                    vertNodeMap[vert_idx] = [node]
                else:
                    vertNodeMap[vert_idx].append(node)
                vert = verts[vert_idx]
                center_2d += Vector2(*vert)
                M.append((*vert, 1))
                b.append((elev[vert_idx]))

            for edge_idx in face_edges:
                edge = tuple(edges[edge_idx])
                if not edge in edgeMap:
                    edgeMap[edge] = [(f, faceObj)]
                elif len(edgeMap[edge]) > 1:
                    raise AttributeError("Edge %s has too many incident faces" % edge_idx)
                else:
                    edgeMap[edge].append((f, faceObj))
            node.center = center_2d / num_verts

            if num_verts == 3:
                # solve explicitly
                try:
                    A, B, C = np.linalg.solve(M, b)
                except np.linalg.linalg.LinAlgError:
                    raise ValueError("Face {} is too close to being co-linear".format(f))
            else:
                # least squares
                x, resid, rank, s = np.linalg.lstsq(M, b, rcond=None)
                # TODO: Use rank and resid to confirm quality of answer:
                #  rank will measure linear independence
                #  resid will report planarity.
                A, B, C = x

            # TODO: This isn't necessarily normalized. If b proves to be the zero vector, then
            # I'm looking at the vector that is the nullspace of the matrix and that's true to
            # arbitrary scale. Confirm that this isn't a problem.
            node.A = A
            node.B = B
            node.C = C
            navMesh.addNode(node)

        print("Found %d edges" % (len(edges)))
        internal = filter(lambda edge: len(edgeMap[tuple(edge)]) > 1, edges)
        external = filter(lambda edge: len(edgeMap[tuple(edge)]) == 1, edges)
        # print("\tFound %d internal edges" % len(list(internal)))
        # print("\tFound %d external edges" % len(list(external)))

        # process the internal edges
        for i, intern in enumerate(internal):
            v0, v1 = intern
            A, B = edgeMap[tuple(intern)]
            a, aFace = A
            b, bFace = B
            na = navMesh.nodes[a]
            na.addEdge(i)
            nb = navMesh.nodes[b]
            nb.addEdge(i)
            edge = Edge()
            edge.v0 = v0
            edge.v1 = v1
            # TODO: Do these two nodes require a particular relationship vis a vis
            # the vertex ordering? I.e., should a be on the left and b on the right?
            # is that even guaranteed?
            edge.n0 = na
            edge.n1 = nb
            navMesh.addEdge(edge)

        # process the external edges (obstacles)
        # for each external edge, make sure the "winding" is opposite that of the face
        obstacles = []
        vertObstMap = {}  # mapping from vertex to the obstacles that are incident to the vertex
        for i, extern in enumerate(external):
            f, face = edgeMap[tuple(extern)][0]
            v0, v1 = extern

            oID = len(obstacles)
            o = Obstacle()
            o.n0 = navMesh.nodes[f]
            if v0 in vertObstMap:
                vertObstMap[v0].append(oID)
            else:
                vertObstMap[v0] = [oID]
            if v1 in vertObstMap:
                vertObstMap[v1].append(oID)
            else:
                vertObstMap[v1] = [oID]

            i0 = face.verts.index(v0)
            vCount = len(face.verts)
            if face.verts[(i0 + 1) % vCount] == v1:
                o.v0 = v0
                o.v1 = v1
            else:
                o.v0 = v1
                o.v1 = v0
            obstacles.append(o)

        processObstacles(obstacles, vertObstMap, vertNodeMap, navMesh)

        print("Found %d obstacles" % len(obstacles))
        #    for o in obstacles:
        #        print '\t', ' '.join( map( lambda x: str(x), o ) )

        navMesh.writeNavFile(self.output['navmesh'], ascii=True)


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
