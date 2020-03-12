import numpy as np
import triangle as tr
from skimage import measure
from typing import List, Tuple


def contour2vertseg(contour: np.ndarray, vertices: List[np.ndarray], segments: List[List[int]], last_idx: int) \
        -> Tuple[List[np.ndarray], List[List[int]], int]:
    """
    convert contours to constraints in format required by triangle package

    :param contour: np.array of contour points
    :param vertices: current list of vertices
    :param segments: current list of segments
    :param last_idx: last index in list of vertices/segments

    :return: updated list of vertices + segments and last_idx
    """
    vertices.extend(contour[:-1])
    verts_in_cnt = len(contour) - 1
    seg = [list(range(last_idx, last_idx + verts_in_cnt)),
           list(range(last_idx + 1, last_idx + verts_in_cnt)) + [last_idx]]
    segments.extend(list(map(list, zip(*seg))))

    last_idx += verts_in_cnt

    return vertices, segments, last_idx


def get_triangles(contours: np.ndarray, bounds: np.ndarray, make_holes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    uses triangle package to triangulate map span by contours
    applies constrained Delaunay triangulation

    :param contours: list of contours that are used as constraints for the triangulation
    :param bounds: same as contours but not used for holes
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

        vertices, segments, last_idx = contour2vertseg(contour, vertices, segments, last_idx)

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

    # also add bounding contours, but do not make them holes
    for bounding_cnt in bounds:
        vertices, segments, last_idx = contour2vertseg(bounding_cnt, vertices, segments, last_idx)
        #
        # if make_holes:
        #     # find point that lies outside contour
        #
        #     # first check corners of enclosing rectangle
        #     mins = np.min(bounding_cnt, axis=0)
        #     maxs = np.max(bounding_cnt, axis=0)
        #     for outside in (mins, maxs, np.array([mins[0], maxs[1]]), np.array([maxs[0], mins[1]])):
        #         out_contour = not measure.points_in_poly(outside.reshape(1, 2), bounding_cnt)[0]
        #         if out_contour:
        #             break
        #     # if corners where not successful sample randomly on map
        #     if not out_contour:
        #         print('random outside point sampling not yet implemented')
        #     holes.append(outside)

    # define contour map as planar straight line graph (pslg) for triangulation
    pslg = {'vertices': np.array(vertices), 'segments': np.array(segments)}

    # add holes if required
    if make_holes and len(holes):
        pslg['holes'] = np.array(holes)

    t = tr.triangulate(pslg, 'pcq')

    triangles = t['triangles']
    vertices = t['vertices']
    traversable_triangles = []
    # remove triangles and vertices that lie outside the bounds
    for bounding_cnt in bounds:
        for triangle in triangles:
            triangle_center = vertices[triangle].mean(axis=0).reshape(1, 2)
            tr_in_bnd = measure.points_in_poly(triangle_center, bounding_cnt)[0]
            if tr_in_bnd:
                traversable_triangles.append(triangle)

    triangles = np.array(traversable_triangles)
    superfluous_vertices = np.setdiff1d(range(len(vertices)), triangles)
    vertices = vertices[np.unique(triangles)]
    # update vertex indices that changed due to removal
    # array 'superfluous_vertices' is already sorted but needs to be reversed to not update wrong indices
    for vert in superfluous_vertices[::-1]:
        triangles[np.where(triangles > vert)] -= 1

    return triangles, vertices


def triangulate_map(contours: np.ndarray, bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param contours: list of arrays of shape (N,2), each array containing a contour of an obstacle with N points
    :param bounds: same format as contours but containing outer boundaries of the traversable area
    :return: tuple (vertices, faces)
        vertices: array of shape (num contour points, 2) containing all vertices (r,c) of the triangle mesh
        faces: array of shape (num edges/3, 3) containing the indices for the vertices that make a triangle (v1,v2,v3)
                for each face of the mesh
    """

    triangles, vertices = get_triangles(contours, bounds)

    faces = []
    for triangle in triangles:
        faces.append(triangle.tolist())

    faces = np.array(faces)

    return vertices, faces
