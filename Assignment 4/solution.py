import numpy as np
import cv2
import math
import random

# ** I TOOK 1 LATE DAY FOR THIS ASSIGNMENT **

def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    # initialize the best set array that we will return
    largest_set = []

    # randomize the matched_pairs list and then cycle through the first 10 random pairs
    randomMatchPairs = random.sample(matched_pairs, len(matched_pairs))
    for i in range(10):
        rndPair = randomMatchPairs[i]

        # get the scale difference
        scaleDiff = keypoints2[rndPair[1], 2] / keypoints1[rndPair[0], 2]

        # get the orientation difference
        # get the orientations of keypoint 1 and 2 for the random pair
        k1Rad = keypoints1[rndPair[0], 3]
        k2Rad = keypoints2[rndPair[1], 3]

        # make sure the orientation is between 0 and 2pi
        k1Rad2pi = k1Rad % (2 * math.pi)
        k2Rad2pi = k2Rad % (2 * math.pi)

        # now convert to degrees
        k1Deg = np.rad2deg(k1Rad2pi)
        k2Deg = np.rad2deg(k2Rad2pi)

        # now calculate orientation difference
        degDiff = abs(k2Deg - k1Deg)
        orientDiff = degDiff
        # orientDiff = min(degDiff, (360-degDiff))

        # make sure they are the same length to avoid indexing problems
        assert(len(randomMatchPairs) == len(matched_pairs))

        # initialize a current set for each iteration
        currSet = []

        for restOfPairsIdx in range(len(matched_pairs)):
            if restOfPairsIdx == i:
                continue
            else:
                # grab a pair from the list
                otherPair = randomMatchPairs[restOfPairsIdx]

                # get the scale difference for otherPair the same way as above
                otherScaleDiff = keypoints2[otherPair[1], 2] / keypoints1[otherPair[0], 2]

                # get the orientation difference for otherPair the same way as above
                otherK1Rad = keypoints1[otherPair[0], 3]
                otherK2Rad = keypoints2[otherPair[1], 3]

                otherK1Rad2pi = otherK1Rad % (2 * math.pi)
                otherK2Rad2pi = otherK2Rad % (2 * math.pi)

                otherK1Deg = np.rad2deg(otherK1Rad2pi)
                otherK2Deg = np.rad2deg(otherK2Rad2pi)

                otherDegDiff = abs(otherK2Deg - otherK1Deg)
                otherOrientDiff = otherDegDiff
                # otherOrientDiff = min(degDiff, (360 - otherDegDiff))

                # check to make sure scale and orientation are within the respective bounds of their agreements,
                # if so then add it to the current set
                upperBoundScale = scaleDiff + (scaleDiff * scale_agreement)
                lowerBoundScale = scaleDiff - (scaleDiff * scale_agreement)
                if (abs(otherOrientDiff - orientDiff) <= orient_agreement and
                        lowerBoundScale <= otherScaleDiff <= upperBoundScale):
                    currSet.append(otherPair)

            # if the current set is bigger than the largest set, then it will become the largest set
            if len(currSet) > len(largest_set):
                largest_set = currSet

    ## END
    assert isinstance(largest_set, list)
    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START

    # need to take the transpose because dot product performs row i * column i
    # so d2 rows need to be converted to columns
    d2T = descriptors2.T

    # take the dot product between the values in the two vectors
    # then perform arccos on the dot product to determine the similarity
    dotProd = np.dot(descriptors1, d2T)
    angleBetweenVectors = np.arccos(dotProd)

    # sort the list so the best values (lower values) will be first
    sortedAngles = np.sort(angleBetweenVectors, axis=1)

    # create a matched_pairs list, check if the best and second-best values are below the threshold,
    # if so then add that pair to the lists
    matched_pairs = []
    for i in range(sortedAngles.shape[0]):
        bestMatch = sortedAngles[i, 0]
        secondBestMatch = sortedAngles[i, 1]
        belowThreshold = (bestMatch / secondBestMatch) <= threshold
        if belowThreshold:
            ogIndexD2 = np.where(angleBetweenVectors == bestMatch)[1][0]
            matched_pairs.append((i, ogIndexD2))

    ## END

    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    xy_points_3d = []
    # xy_points is a 2d matrix and needs to be converted to 3d
    for i in range(xy_points.shape[0]):
        xy_points_3d = np.insert(xy_points, 2, 1, axis=1)

    assert xy_points_3d.shape == (xy_points.shape[0], 3)

    # need to transpose the points array to make it into the correct shape to apply h
    xy_points_3d_T = xy_points_3d.T

    # apply the homography matrix to the points
    hom_xy_points1 = np.matmul(h, xy_points_3d_T)

    # convert back to original shape
    hom_xyz_points = hom_xy_points1.T

    # replace all instances of zero with an arbitrarily small number for when we divide
    hom_xyz_points[hom_xyz_points == 0] = 1e-10

    # extract the last column, so I can divide with it
    zList = np.array([(hom_xyz_points[i, 2]) for i in range(hom_xyz_points.shape[0])])
    zListT = zList.reshape((hom_xyz_points.shape[0], 1))

    # divide by the extra dimension, get rid of extra column, and return points
    xy_points_out = hom_xyz_points / zListT
    xy_points_out = np.delete(xy_points_out, 2, 1)

    # END
    return xy_points_out


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keypoint xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keypoints are within a `tol`
    radius to the corresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol * 1.0

    # START
    best_set = []
    h = np.ones((3, 3)) #stub

    for i in range(num_iter):
        # initialize a current set that we will add good points into
        curr_set = []

        # select 4 random pairs and make sure there is no duplicates
        randomInts = random.sample(range(0, (xy_src.shape[0]-1)), 4)
        randomSrcPoints = np.array([[xy_src[randomInts[0]]], [xy_src[randomInts[1]]], [xy_src[randomInts[2]]],
                                   [xy_src[randomInts[3]]]])
        randomRefPoints = np.array([[xy_ref[randomInts[0]]], [xy_ref[randomInts[1]]], [xy_ref[randomInts[2]]],
                                   [xy_ref[randomInts[3]]]])

        # find the homography matrix between src and ref point, and use matrix to project src points
        currH, mask = cv2.findHomography(randomSrcPoints, randomRefPoints)
        xyRefProj = KeypointProjection(xy_src, currH)

        # loop through all projected points and check if the euclidean distance between projected point and reference
        # point is no more than tol away, if it is less, then append to curr_set
        for j in range(xy_src.shape[0]):
            projRefPoint = xyRefProj[j]
            currRefPoint = xy_ref[j]
            if math.dist(currRefPoint, projRefPoint) <= tol:
                curr_set.append(projRefPoint)

        # if the curr_set is bigger than the best_set than it will become the new best set and save corresponding h
        if len(curr_set) > len(best_set):
            best_set = curr_set
            h = currH


    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac

# %%
