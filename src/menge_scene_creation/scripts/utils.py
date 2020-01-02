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


def get_triangles(contours):
    vertices = []
    segments = []
    # holes = []

    last_idx = 0
    # extract vertices and edges (segments) from contours
    for contour in contours:
        vertices.extend(contour[:-1])
        verts_in_cnt = len(contour) - 1
        seg = [list(range(last_idx, last_idx + verts_in_cnt)),
               list(range(last_idx + 1, last_idx + verts_in_cnt)) + [last_idx]]
        segments.extend(list(map(list, zip(*seg))))

        last_idx += verts_in_cnt

    # define contour map as planar straight line graph (pslg) for triangulation
    pslg = {'vertices': np.array(vertices), 'segments': np.array(segments)}  # , 'holes': []}

    t = tr.triangulate(pslg, 'pc')

    triangles = t['triangles']
    vertices = t['vertices']
    return triangles, vertices


def triangulate_map(shape, contours, obstacle_height):
    """

    :param shape: (r, c) dimensions of the total map
    :param contours: list of arrays of shape (N,2), each array containing a contour of an obstacle with N points
    :param obstacle_height: float defining the height of the obstacles

    :return: tuple (vertices, edges, faces, elevation)
        vertices: array of shape (num contour points, 2) containing all vertices (r,c) of the triangle mesh
        edges: array of shape (num contour points, 2) containing the indices that make an edge (e1,e2)
                for each edge of the mesh
        faces: array of shape (num edges/3, 3) containing the indices for the vertices that make a triangle (v1,v2,v3)
                for each face of the mesh
        elevation: array of shape (num contour points,) defining the height (z-coord.) of each vertex
    """

    triangles, vertices = get_triangles(contours)

    # initialize elevation for each vertex to zero
    elevation = np.zeros(len(vertices))
    # connect every combination of the triangle's vertices to edges and add all three edges as a face
    edges = []
    faces = []
    faces_edges = []
    for triangle in triangles:
        face = {'verts': [], 'edges': []}
        face_edges = []
        # # make sure triangle winding is the same for all triangles:
        # triangle_verts = vertices[triangle]
        # triangle_verts = np.c_[triangle_verts, np.zeros(len(triangle_verts))]
        # z = np.cross(triangle_verts[1] - triangle_verts[0], triangle_verts[2] - triangle_verts[0])[2]
        # if z < 0:
        #     triangle[1], triangle[2] = triangle[2], triangle[1]

        face['verts'] = triangle.tolist()
        triangle_edges = [[triangle[0], triangle[1]],
                          [triangle[1], triangle[2]],
                          [triangle[2], triangle[0]]]
        for edge in triangle_edges:
            if edges:
                edge_idx = np.argwhere(np.all(np.isin(edges, edge), axis=1)).ravel().tolist()
            else:
                edge_idx = []
            if not edge_idx:
                edges.append(sorted(edge))
                edge_idx = [len(edges) - 1]
            face['edges'].extend(edge_idx)
            face_edges.extend(edge_idx)
        faces.append(face)
        faces_edges.append(face_edges)

    # for each obstacle contour --> elevate to obstacle height
    for cnt_idx, contour in enumerate(contours):
        # find vertices that lie inside of contour
        verts_in_contour_idx = measure.points_in_poly(vertices, contour)
        # function also returns some points that lie on contour --> filter these points out
        real_inner = np.logical_not(np.all(np.isin(vertices[verts_in_contour_idx], contour), axis=1))
        verts_in_contour = vertices[verts_in_contour_idx][real_inner]
        verts_in_contour_idx = np.all(np.isin(vertices, verts_in_contour), axis=1)
        # if vertex is inside of contour --> set to obstacle height
        elevation[verts_in_contour_idx] = obstacle_height

        # append all unique contour points to the set of vertices
        vertices = np.append(vertices, contour[:-1], axis=0)
        # make those new contour points to obstacle height
        elevation = np.append(elevation, [obstacle_height] * (len(contour) - 1))
        ###
        # find indices of the vertices that belong to the contour
        for point, nextpoint in zip(contour[:-1], contour[1:]):
            pnt_idx = np.argwhere(np.all(np.isin(vertices, point), axis=1)).ravel()
            # assert len(pnt_idx) == 2, '\ncontour no. %s\n' \
            #                           'point %s\n' \
            #                           'point idx %s\n' \
            #                           'next point %s\n' \
            #                           'next point idx %s\n' % (cnt_idx, point, pnt_idx, nextpoint, next_pnt_idx)
            next_pnt_idx = np.argwhere(np.all(np.isin(vertices, nextpoint), axis=1)).ravel()
            # assert len(next_pnt_idx) == 2, '\ncontour no. %s\n' \
            #                                'point %s\n' \
            #                                'point idx %s\n' \
            #                                'next point %s\n' \
            #                                'next point idx %s\n' % (cnt_idx, point, pnt_idx, nextpoint, next_pnt_idx)
            lower_edge = np.argwhere(np.all(np.isin(edges, [pnt_idx[0], next_pnt_idx[0]]), axis=1)).ravel()[0]
            edges.append(sorted([next_pnt_idx[0], next_pnt_idx[1]]))
            edges.append(sorted([pnt_idx[1], next_pnt_idx[1]]))
            edges.append(sorted([pnt_idx[0], pnt_idx[1]]))
            upper_edge = len(edges) - 2
            where_lower = np.argwhere(np.any(np.isin(faces_edges, lower_edge), axis=1)).ravel().tolist()
            assert len(where_lower) <= 2, 'more than two faces share same edge'
            for face in where_lower:
                # if center of face lies within contour --> replace lower edge with upper edge
                if measure.points_in_poly(vertices[faces[face]['verts']].mean(axis=0).reshape(1, 2), contour)[0]:
                    for i, edge in enumerate(faces[face]['edges']):
                        if edge == lower_edge:
                            faces[face]['edges'][i] = upper_edge
                            break
                    break
            face_edges = [lower_edge]
            face_edges.extend(list(range(len(edges) - 3, len(edges))))
            # face_verts_idx = [*next_pnt_idx, *pnt_idx]
            # face_verts = vertices[face_verts_idx]
            face_verts_idx = [pnt_idx[0], next_pnt_idx[0], next_pnt_idx[1], pnt_idx[1]]
            faces.append({'verts': face_verts_idx, 'edges': face_edges})

    edges = np.array(edges)
    return vertices, edges, faces, elevation
