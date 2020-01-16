import numpy as np
import triangle as tr
from skimage import measure


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
        faces: array of shape (num edges/3, 3) containing the indices for the vertices that make a triangle (v1,v2,v3)
                for each face of the mesh
    """

    triangles, vertices = get_triangles(contours)

    faces = []
    for triangle in triangles:
        faces.append(triangle.tolist())

    faces = np.array(faces)

    return vertices, faces