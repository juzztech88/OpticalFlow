"""System module."""
import cv2
import numpy as np


# Parameters for Shi-Tomasi corner detection
FEATURE_PARAMS = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def create_video_capture_writer(src_vid, dst_vid, codec='mp4v'):
    """ Create a video capture and a video write objects

    :param src_vid:     (str) Path of source video
    :param dst_vid:     (str) Path of output video
    :param codec:       (str) Four-cc code of codec type
    :return:
        cap:    Video capture object
        out:    Video writer object
    """

    cap = cv2.VideoCapture(src_vid)  # Instantiate the video capturing object
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get width of source video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get height of source video
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate of source video
    else:
        raise ConnectionError("OpenCV cannot open the source video.")

    # Specify video codec with FourCC 4-byte code
    # https://www.fourcc.org/codecs.php
    # 'MJPG' works with Colab
    fourcc = cv2.VideoWriter_fourcc(*codec)  # Instantiate a video codec object

    # Initialize video write object
    # cv2.VideoWriter(filename, fourcc, fps, frameSize)
    # https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5
    out = cv2.VideoWriter(dst_vid, fourcc, fps, (width, height))  # Instantiate a video write object

    return cap, out


def of_vec_magnitude_direction(origin_arr, dest_arr):
    """ Compute the optical flow vectors.

    :param origin_arr:      (Numpy array) An array of the origin of every optical flow vector.
    :param dest_arr:        (Numpy array) An array of the destination of every optical flow vector.
    :return:
        of_vec:             (Numpy array) An array of optical flow represented as (vector_magnitude, vector_direction)
    """
    of_vec = dest_arr - origin_arr
    of_vec_mag = np.linalg.norm(of_vec, ord=None, axis=1)

    of_vec_x, of_vec_y = of_vec[:, 0], of_vec[:, 1]
    of_vec_dir = np.arctan2(of_vec_x, of_vec_y) * 180 / np.pi

    return np.stack((of_vec_mag, of_vec_dir), axis=1)


def sparse_v1(src_vid, dst_vid, codec='mp4v'):
    """  Calculate the sparse optical flow of a source video using Lucas-Kanade
    optical flow.

    Code adapted from:
    1. https://code.luasoftware.com/tutorials/jupyter/display-opencv-video-in-jupyter/
    2. https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    :param src_vid:     (str) Path of source video
    :param dst_vid:     (str) Path of output video
    :param codec:       (str) Four-cc code of codec type
    """

    cap, out = create_video_capture_writer(src_vid, dst_vid, codec=codec)

    # Create random colors to draw the tracks of optical flow
    color = np.random.randint(0, 255, (100, 3))

    # Take the first frame and find corners in it
    read_ok, old_frame = cap.read()
    if not read_ok:
        raise IOError("OpenCV cannot read the first frame.")

    # Convert first frame to grayscale as starting frame of optical flow computation
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Get the Shi-Tomasi corner points as the starting set of sparse feature points
    p_0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **FEATURE_PARAMS)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Looping over video frames to compute stream of optical flow
    while True:
        read_ok, new_frame = cap.read()   # grab the next video frame
        if not read_ok:
            break

        # Convert the next frame to grayscale
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p_1, s_t, __ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p_0, None, **LK_PARAMS)

        # Select good points
        if p_1 is not None:
            good_new = p_1[s_t == 1]
            good_old = p_0[s_t == 1]
        else:
            break

        # Draw the infinitesimal tracks for all the good points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            new_ix, new_iy = new.ravel()
            old_ix, old_iy = old.ravel()

            # a black mask with tracks
            mask = cv2.line(mask, (int(new_ix), int(new_iy)), (int(old_ix), int(old_iy)),
                            color[i].tolist(), 2)

            # draw circular dots at the new good points on the new frame
            new_frame = cv2.circle(new_frame, (int(new_ix), int(new_iy)), 5,
                                   color[i].tolist(), -1)

        # Combine the mask and the new frame as frame of output video
        dst_frame = cv2.add(new_frame, mask)
        out.write(dst_frame)

        # Update the old frame and features
        old_gray = new_gray.copy()
        p_0 = good_new.reshape(-1, 1, 2)

    out.release()
    cap.release()


def sparse_with_statistics_v1(src_vid, dst_vid, codec='mp4v', min_ft_pts=50):
    """ Calculate the sparse optical flow while keeping track of the flow vectors.\n

    Steps:\n
    1.  Read the first video frame. Convert the frame to grayscale.\n
    2.  Detect Shi-Tomasi feature points of the first video frame.\n
    3.  Initialize the array of optical flow vectors as a (n, 2) zero array,
        where n represents the total Shi-Tomasi feature points detected.\n
    4.  Also initialize an empty list to keep track of the sequence of flow vectors.
    4.  Read the next video frame. Convert the frame to grayscale.\n
    5.  Compute the Lucas Kanade optical flow vectors between the first and
        the next frame in grayscale.\n
    6.  Retain only the optical flow vectors of the good tracking points
    7.  Convert the flow vectors from (u, v)-representation to
        (magnitude, angle) representation.\n
    8.  Save the array of flow vectors in the tracking list.\n
    9.  If the number of good tracking points is below the minimum threshold,
        re-compute the Shi-Tomasi feature points on the second frame.\n
    10. Repeat steps 1 - 9 by iterating over the video frames and updating
        the feature points of the previous frame to those of the next frame
        in each iteration.\n

    :param src_vid:     (str) Path of source video
    :param dst_vid:     (str) Path of output video
    :param codec:       (str) Four-cc code of codec type
    :param min_ft_pts:  (int) Minimum threshold for the total good tracking points
    """

    # Create the video capture and video writer objects
    cap, out = create_video_capture_writer(src_vid, dst_vid, codec=codec)

    # Initialization the current video frame
    read_ok, current_frame = cap.read()
    if not read_ok:
        raise IOError('OpenCV cannot read the video frame.')
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the Shi-Tomasi feature points
    current_ft_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **FEATURE_PARAMS)
    # Get the number of Shi-Tomasi feature points; Shape of current_ft_pts is [n_pts, 1, 2]
    n_ft_pts = current_ft_pts.shape[0]

    # Initialize the list that stores the array of optical flow vectors for each frame iteration
    all_of_vec = [np.zeros((n_ft_pts, 2))]

    # Initialize a random drawing color
    color = np.random.randint(0, 255, (3,)).tolist()

    # Draw circles around all the feature points on top of the first frame as first output frame
    current_output_frame = current_frame.copy()
    for ft_pt in current_ft_pts:
        ft_pt_x, ft_pt_y = ft_pt.ravel()
        current_output_frame = cv2.circle(current_output_frame, (int(ft_pt_x), int(ft_pt_y)), 5,
                                          color, -1)
    out.write(current_output_frame)

    # Initialize a mask for creating optical flow trajectory
    mask = np.zeros_like(current_frame)

    while True:
        read_ok, next_frame = cap.read()
        if not read_ok:
            break   # Break out of while loop when there is no more frame to iterate

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        next_ft_pts, ft_pts_status, __ = cv2.calcOpticalFlowPyrLK(
            current_gray, next_gray, current_ft_pts, None, **LK_PARAMS
        )   # Calculate optical flow

        if next_ft_pts is None:
            current_ft_pts = cv2.goodFeaturesToTrack(next_gray, mask=None, **FEATURE_PARAMS)
            n_ft_pts, __ = current_ft_pts.shape
            next_of_vec = np.zeros((n_ft_pts, 2))
            all_of_vec.append(next_of_vec)
            current_gray = next_gray.copy()
            mask = np.zeros_like(next_frame)    # Restart the mask
            color = np.random.randint(0, 255, (3,)).tolist()    # Restart the drawing color
            out.write(next_frame)   # Add original frame to output
            continue

        # Get the sets of good tracking points, compute and save their optical flow vectors
        good_current_ft_pts = current_ft_pts[ft_pts_status == 1]
        good_next_ft_pts = next_ft_pts[ft_pts_status == 1]
        next_of_vec = of_vec_magnitude_direction(good_current_ft_pts, good_next_ft_pts)
        all_of_vec.append(next_of_vec)

        # Generate the next output video frame
        for current_ft_pt, next_ft_pt in zip(good_current_ft_pts, good_next_ft_pts):
            current_x, current_y = current_ft_pt.ravel()
            next_x, next_y = next_ft_pt.ravel()
            # draw the optical flow vector line on the mask
            mask = cv2.line(mask, (int(current_x), int(current_y)), (int(next_x), int(next_y)),
                            color, 2)
            # draw circle around the endpoint of optical flow vectors on the next output frame
            next_frame = cv2.circle(next_frame, (int(next_x), int(next_y)), 5, color, -1)

        next_frame = cv2.add(next_frame, mask)
        out.write(next_frame)

        if np.sum(ft_pts_status) < min_ft_pts:
            current_ft_pts = cv2.goodFeaturesToTrack(next_gray, mask=None, **FEATURE_PARAMS)
            color = np.random.randint(0, 255, (3,)).tolist()
            mask = np.zeros_like(next_frame)
        else:
            current_ft_pts = next_ft_pts

        # Updates grayscale image for computation of optical flow in next iteration
        current_gray = next_gray.copy()

    cap.release()
    out.release()
    return all_of_vec
