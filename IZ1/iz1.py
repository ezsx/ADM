import cv2
import numpy as np


def initialize_tracker(tracker_type, frame, bbox):
    if tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    else:
        raise ValueError("Unknown tracker type: {}".format(tracker_type))


def track_with_opencv_tracker(video_path, tracker_type, bbox):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Could not open video file: " + video_path)

    ok, frame = video.read()
    if not ok:
        raise IOError("Could not read the first frame from video file: " + video_path)

    tracker = initialize_tracker(tracker_type, frame, bbox)
    ok = tracker.init(frame, bbox)
    if not ok:
        raise Exception("Tracker initialization failed.")

    tracked_bboxes = []
    frames = []

    total_frames = 0
    lost_frames = 0
    total_distance = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        total_frames += 1
        ok, new_bbox = tracker.update(frame)

        if ok:
            x, y, w, h = [int(v) for v in new_bbox]
            tracked_bboxes.append(new_bbox)
            frames.append(frame)

            # Calculate the true center using new_bbox
            true_center = (x + w / 2, y + h / 2)
            tracked_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            total_distance += np.sqrt((true_center[0] - tracked_center[0]) ** 2 +
                                      (true_center[1] - tracked_center[1]) ** 2)
        else:
            lost_frames += 1
            tracked_bboxes.append(tracked_bboxes[-1])

    print(f"{tracker_type} tracking: Average distance: {total_distance / total_frames}")
    print(f"{tracker_type} tracking: Lost frames: {lost_frames}")
    print(f"{tracker_type} tracking: Total frames: {total_frames}")

    return tracked_bboxes, frames


def initialize_template_tracker(frame, bbox):
    x, y, w, h = bbox
    template = frame[y:y + h, x:x + w]
    return template, bbox


def match_template(frame, template):
    method = cv2.TM_CCOEFF_NORMED
    res = cv2.matchTemplate(frame, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.8:
        return None

    top_left = max_loc
    h, w = template.shape[:2]
    return top_left[0], top_left[1], w, h


def update_template(frame, bbox):
    x, y, w, h = bbox
    return frame[y:y + h, x:x + w]


def track_with_template_matching(video_path, bbox):
    """
    Initial Template Extraction:

    1) On the first frame, extract a template based on the provided bounding box. This template is a sub-image of the object you want to track.
    Template Matching in Subsequent Frames:

    2) For each new frame, search for a region that best matches the template. This is done using a method like normalized cross-correlation.
    Adaptive Template Update:

    3) If a good match is found, update the template with the newly matched region. This helps to adapt to changes in the object's appearance over time.
    Calculating Match Quality:

    4) Check the quality of the match. If it falls below a threshold (indicating a poor match), do not update the template.
    Handling Lost Tracking:

    5) If the object is lost (i.e., no part of the frame sufficiently matches the template), continue tracking with the last known good template.
    Output Tracking Data:

    6) For each frame, save or output the location of the matched region, which represents the tracked position of the object.

    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Could not open video file: " + video_path)

    ok, frame = video.read()
    if not ok:
        raise IOError("Could not read the first frame from video file: " + video_path)

    template, bbox = initialize_template_tracker(frame, bbox)
    tracked_bboxes = [bbox]
    frames = [frame]

    total_frames = 0
    lost_frames = 0
    total_distance = 0

    while True:
        ok, frame = video.read()

        if not ok:  # Break the loop if no more frames are available
            break

        total_frames += 1
        new_bbox = match_template(frame, template)
        if new_bbox is not None:
            x, y, w, h = [int(v) for v in new_bbox]
            tracked_bboxes.append(new_bbox)
            frames.append(frame)
            template = update_template(frame, new_bbox)

            # Calculate the true center using new_bbox
            true_center = (x + w / 2, y + h / 2)
        else:
            # If tracking is lost, use the last known good bounding box
            tracked_bboxes.append(tracked_bboxes[-1])
            true_center = None

        if true_center:
            # Calculate the distance between true center and tracked center
            tracked_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            total_distance += np.sqrt((true_center[0] - tracked_center[0]) ** 2 +
                                      (true_center[1] - tracked_center[1]) ** 2)
        else:
            lost_frames += 1

    print("Template matching: Average distance: {}".format(total_distance / total_frames))
    print("Template matching: Lost frames: {}".format(lost_frames))
    print("Template matching: Total frames: {}".format(total_frames))
    return tracked_bboxes, frames


def show_video():
    # Paths to your videos and initial bounding boxes
    # video_paths = ['vids/IMG_3025.MOV', 'vids/IMG_3026.MOV', 'vids/IMG_3027.MOV', 'vids/IMG_3028.MOV',
    video_paths = ['vids/IMG_3029.MOV']
    initial_bboxes = [(120, 120, 170, 170), (500, 100, 170, 170), (490, 500, 170, 170), (100, 500, 170, 170),
                      (300, 150, 170, 170)]
    trackers = ['KCF', 'CSRT', 'MOSSE', 'Template']

    for video_file_path, initial_bbox in zip(video_paths, initial_bboxes):
        for tracker in trackers:
            print(f"Running {tracker} tracker on {video_file_path}...")
            if tracker != 'Template':
                bboxes, frames = track_with_opencv_tracker(video_file_path, tracker, initial_bbox)
            else:
                bboxes, frames = track_with_template_matching(video_file_path, initial_bbox)

            # Optionally display the results
            for i, frame in enumerate(frames):
                x, y, w, h = [int(v) for v in bboxes[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow(tracker, frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()


def save_all():
    import os

    # Paths to your videos
    video_paths = ['vids/IMG_3025.MOV', 'vids/IMG_3026.MOV', 'vids/IMG_3027.MOV', 'vids/IMG_3028.MOV',
                   'vids/IMG_3029.MOV']
    trackers = ['KCF', 'CSRT', 'MOSSE', 'Template']
    initial_bboxes = [(120, 120, 170, 170), (500, 100, 170, 170), (490, 500, 170, 170)
        , (100, 500, 170, 170), (300, 150, 170, 170)]

    for video_file_path, initial_bbox in zip(video_paths, initial_bboxes):
        # Create a directory for each video's results
        result_dir = video_file_path.split('.')[0]
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for tracker in trackers:
            if tracker != 'Template':
                bboxes, frames = track_with_opencv_tracker(video_file_path, tracker, initial_bbox)
            else:
                bboxes, frames = track_with_template_matching(video_file_path, initial_bbox)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(result_dir, f"{tracker}_output.mp4"), fourcc, 20.0,
                                  (frames[0].shape[1], frames[0].shape[0]))

            # Write frames to video
            for i, frame in enumerate(frames):
                # Draw bounding box on frame
                x, y, w, h = [int(v) for v in bboxes[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Write the frame
                out.write(frame)

            # Release the video writer
            out.release()


# save_all()
show_video()
