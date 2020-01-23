import numpy as np
from skimage import measure


def remove_inner_contours(contours, trajectory):
    """
    removes all contours that lie within other contours

    :param contours:            list of contours specified as numpy arrays (n,2)
    :param trajectory:          array of points belonging to the traversable space

    :return:
        - reduced_contours:  list of contours where inner contours are removed
        - bounding_contours: list of contours that enclose the traversable space
    """
    inner_contours = []
    for idx1, contour1 in enumerate(contours):
        inner_contours.append([])
        for idx2, contour2 in enumerate(contours):
            if not np.array_equal(contour1, contour2):
                # get mask that tells which points of contour2 lie within contour1
                cnt2_in_cnt1 = measure.points_in_poly(contour2, contour1)
                if np.all(cnt2_in_cnt1):
                    # if all points of one contours lie within another, add to list of inner contours
                    inner_contours[idx1].extend([idx2])

    # check for contours that enclose the traversable space/ trajectory
    inner_cnt_idx = 0
    num_contours = len(inner_contours)
    bounding_contours = []
    for cnt_idx, contour in enumerate(contours):
        # if the trajectory lies within a contour --> this contour is a bounding contour of the traversable area
        # get mask that tells which points of trajectory lie within contour
        trajectory_in_contour = measure.points_in_poly(trajectory, contour)
        if np.all(trajectory_in_contour):
            # other contours within the bounding contour are obstacles and must not be deleted
            inner_contours.remove(inner_contours[inner_cnt_idx])
            bounding_contours.append(cnt_idx)
        else:
            inner_cnt_idx += 1

    # only keep innermost bounding contour
    real_bound = np.array([])
    if len(bounding_contours) > 1:
        for bound1 in bounding_contours:
            cnt_bnd1 = contours[bound1]
            for bound2 in bounding_contours:
                cnt_bnd2 = contours[bound2]
                if not np.array_equal(cnt_bnd1, cnt_bnd2):
                    # get mask that tells which points of bound2 lie within bound1
                    bn2_in_bnd1 = measure.points_in_poly(cnt_bnd2, cnt_bnd1)
                    if not np.any(bn2_in_bnd1):
                        real_bound = [bound1]
                        # Break the inner loop..
                        break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break
    # if only one bounding contour, or multiple intersecting bounding contours
    if not real_bound:
        # return all inner contours, sorted ascending by the area they enclose
        contour_area = lambda cnt: np.abs(0.5*np.sum(cnt[:, 1][:-1]*np.diff(cnt[:, 0]) - cnt[:, 0][:-1]*np.diff(cnt[:, 1])))
        order = np.argsort([contour_area(cnt) for cnt in bounding_contours])
        real_bound = bounding_contours[order]

    # make inner_contours (that should be removed) unique
    to_remove = set(np.concatenate(inner_contours)).union(bounding_contours)
    # delete inner contours
    reduced_contours = list(np.delete(np.array(contours), list(to_remove)))
    # reverse order of contour vertices to maintain correct winding
    bounding_contours = [contours[cnt_idx] for cnt_idx in real_bound]
    # bounding_contours = [np.flip(contours[cnt_idx], axis=0) for cnt_idx in real_bound]

    return reduced_contours, bounding_contours


def contour_length(cnt):
    """
    get length of contour
    """
    diff = cnt - np.roll(cnt, 1, axis=0)
    return np.sum(np.linalg.norm(diff, axis=1))


def lineContour2rectangle(cnt_length, cnt, tol):
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


def approximate_contours(contours, tolerance):
    """
    approximates each cnt within contours with a polygon given the specified tol

    :param contours:        list of contours specified as numpy arrays (n,2)
    :param tolerance:       float, maximum distance from original points of polygon to approximated polygonal chain

    :return:                list of approximated contours
    """

    approx_contours = []
    for contour in contours:
        # make sure cnt is closed
        if not np.array_equiv(contour[0], contour[-1]):
            contour = np.append(contour, contour[0]).reshape(-1, 2)
        # only polygons of higher order than triangles need to be approximated
        if len(contour) > 3:
            approx = measure.approximate_polygon(contour, tolerance)
            # remove obstacles smaller than the tol
            if contour_length(approx) > tolerance:
                if len(approx) > 3:
                    approx_contours.append(approx)
                else:
                    # inflate contours that are only lines to rectangle
                    contour = lineContour2rectangle(contour_length, approx, tolerance)
                    approx_contours.append(contour)
        # if extracted contour only line --> filter too small contours
        elif contour_length(contour) >= tolerance / 2:
            # inflate contours that are only lines to rectangle
            contour = lineContour2rectangle(contour_length, contour, tolerance)
            approx_contours.append(contour)

    return approx_contours
