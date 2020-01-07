from xml.etree import ElementTree as ET
import yaml
import numpy as np
from skimage import measure
import cv2
import triangle as tr


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
    approximates each cnt within contours with a polygon given the specified tol

    :param contours:        list of contours specified as numpy arrays (n,2)
    :param tolerance:       float, maximum distance from original points of polygon to approximated polygonal chain

    :return:                list of approximated contours
    """

    def cnt_length(cnt):
        """
        get lenght of contour
        """
        diff = cnt - np.roll(cnt, 1, axis=0)
        return np.sum(np.linalg.norm(diff, axis=1))

    def lineContour2rectangle(cnt, tol):
        """
        make two point contour line to (closed) rectangle contour
        """
        len_cnt = cnt_length(cnt)
        # such that rectangle perimeter >= tol and each side at least 1 pixel long
        offset = max(1, (tol - len_cnt) / 4)
        # compute the normal vector to the contour line
        vec_diff = cnt[1] - cnt[0]
        vect_normal = np.array([vec_diff[1], -vec_diff[0]])
        norm_vect_normal = vect_normal / np.linalg.norm(vect_normal)
        # move contour point [0] and [1] each by +/- offset in direction of norm_vect_normal
        offset_vect = offset * norm_vect_normal
        # order points such that contour is still counter-clockwise and closed
        cnt = np.array([cnt[0] + offset_vect, cnt[1] + offset_vect,
                        cnt[1] - offset_vect, cnt[0] - offset_vect,
                        cnt[0] + offset_vect])
        return cnt

    approx_contours = []
    for contour in contours:
        # make sure cnt is closed
        if not np.array_equiv(contour[0], contour[-1]):
            contour = np.append(contour, contour[0]).reshape(-1, 2)
        # only polygons of higher order than triangles need to be approximated
        if len(contour) > 3:
            approx = measure.approximate_polygon(contour, tolerance)
            # remove obstacles smaller than the tol
            if cnt_length(approx) > tolerance:
                if len(approx) > 3:
                    approx_contours.append(approx)
                else:
                    # inflate contours that are only lines to rectangle
                    contour = lineContour2rectangle(approx, tolerance)
                    approx_contours.append(contour)
        # if extracted contour only line --> filter too small contours
        elif cnt_length(contour) >= tolerance / 2:
            # inflate contours that are only lines to rectangle
            contour = lineContour2rectangle(contour, tolerance)
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
    pivot_x = center_x - cos_a * width / 2 + sin_a * height / 2
    pivot_y = center_y - cos_a * height / 2 - sin_a * width / 2

    return (pivot_x, pivot_y), (width, height), angle


# def asvoid(arr):
#     """
#     Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
#     View the array as dtype np.void (bytes). The items along the last axis are
#     viewed as one value. This allows comparisons to be performed on the entire row.
#     """
#     arr = np.ascontiguousarray(arr)
#     if np.issubdtype(arr.dtype, np.floating):
#         """ Care needs to be taken here since
#         np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
#         Adding 0. converts -0. to 0.
#         """
#         arr += 0.
#     return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))
#
#
# def inNd(a, b, assume_unique=False):
#     """
#     works like numpy in1d but for n-dimensional arrays by
#     """
#     a = asvoid(a)
#     b = asvoid(b)
#     return np.in1d(a, b, assume_unique)


def get_triangles(contours, make_holes=True):
    """
    uses triangle package to triangulate map span by contours
    applies constrained Delaunay triangulation

    :param contours: list of contours that are used as constraints for the triangulation
    :param make_holes: if True, does not triangulate inside each contour
    :return:
        triangles: triangles defined via the indices for the corresponding vertices
        vertices: x,y for each vertex
    """
    vertices = []
    segments = []
    holes = []

    last_idx = 0
    # extract vertices and edges (segments) from contours
    for contour in contours:
        vertices.extend(contour[:-1])
        verts_in_cnt = len(contour) - 1
        seg = [list(range(last_idx, last_idx + verts_in_cnt)),
               list(range(last_idx + 1, last_idx + verts_in_cnt)) + [last_idx]]
        segments.extend(list(map(list, zip(*seg))))

        last_idx += verts_in_cnt

        if make_holes:
            # find point that lies inside of contour

            # first check whether mean of contour points lies within contour
            center = contour.mean(axis=0)
            in_contour = measure.points_in_poly(center.reshape(1, 2), contour)[0]
            idx = 0
            while not in_contour:
                # form triangle of 3 consecutive contour points
                part_triangle = contour[idx:idx+3]
                # if triangle lies within contour --> center point is within contour
                center = part_triangle.mean(axis=0)
                # check if center lies within contour
                in_contour = measure.points_in_poly(center.reshape(1, 2), contour)[0]
                idx += 1
                if idx > len(contour):
                    raise ValueError("Unable to find point in contour")
            # for each contour, append point that lies within it
            holes.append(center)

    # define contour map as planar straight line graph (pslg) for triangulation
    pslg = {'vertices': np.array(vertices), 'segments': np.array(segments)}

    # add holes if required
    if make_holes:
        pslg['holes'] = np.array(holes)

    t = tr.triangulate(pslg, 'pc')

    triangles = t['triangles']
    vertices = t['vertices']
    return triangles, vertices


def triangulate_map(contours):
    """

    :param contours: list of arrays of shape (N,2), each array containing a contour of an obstacle with N points

    :return: tuple (vertices, edges, faces, elevation)
        vertices: array of shape (num contour points, 2) containing all vertices (r,c) of the triangle mesh
        edges: array of shape (num contour points, 2) containing the indices that make an edge (e1,e2)
                for each edge of the mesh
        faces: array of shape (num edges/3, 3) containing the indices for the vertices that make a triangle (v1,v2,v3)
                for each face of the mesh
        elevation: array of shape (num contour points,) defining the height (z-coord.) of each vertex
    """

    triangles, vertices = get_triangles(contours)

    # set elevation for each vertex to zero
    elevation = np.zeros(len(vertices))

    edges = []
    faces = []
    faces_verts = []
    faces_edges = []
    for triangle in triangles:

        face_edges = []
        # # make sure triangle winding is the same for all triangles:
        # triangle_verts = vertices[triangle]
        # triangle_verts = np.c_[triangle_verts, np.zeros(len(triangle_verts))]
        # z = np.cross(triangle_verts[1] - triangle_verts[0], triangle_verts[2] - triangle_verts[0])[2]
        # if z < 0:
        #     triangle[1], triangle[2] = triangle[2], triangle[1]

        faces_verts.append(triangle.tolist())

        # connect every combination of the triangle's vertices to edges and add all three edges as a face
        triangle_edges = [[triangle[0], triangle[1]],
                          [triangle[1], triangle[2]],
                          [triangle[2], triangle[0]]]
        for edge in triangle_edges:
            if edges:
                # to only have unique edges -> find edge in list of edges and return index
                edge_idx = np.argwhere(np.all(np.isin(edges, edge), axis=1)).ravel().tolist()
            else:
                edge_idx = []
            # if edge is not yet in list of edges, add it
            if not edge_idx:
                edges.append(sorted(edge))
                edge_idx = [len(edges) - 1]
            face_edges.extend(edge_idx)
        faces_edges.append(face_edges)

    faces_verts = np.array(faces_verts)
    faces_edges = np.array(faces_edges)

    # remove all vertices that became obsolete by skipping triangles that are inside contours
    verts_obsolete = []
    for v, vertex in enumerate(vertices):
        if not np.any(np.isin(faces_verts, v)):
            verts_obsolete.append(v)
            # adjust vertex indices
            faces_verts[np.greater(faces_verts, v)] -= 1

    vertices = np.delete(vertices, verts_obsolete, axis=0)

    # each face is defined by its vertices and edges
    for i, verts in enumerate(faces_verts):
        face = {'verts': verts, 'edges': faces_edges[i]}
        faces.append(face)

    edges = np.array(edges)
    return vertices, edges, faces, elevation
