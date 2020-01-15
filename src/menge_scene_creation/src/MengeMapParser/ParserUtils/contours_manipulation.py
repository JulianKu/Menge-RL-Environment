import numpy as np
from skimage import measure


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
