from collections import deque
import lane_detection_functions

QUEUE_LENGTH=50

class LaneDetector:
#    def __init__(self):
#        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
#        self.right_lines = deque(maxlen=QUEUE_LENGTH)

    def process(self, image):
        white_yellow = lane_detection_functions.select_white_yellow(image)
        gray         = lane_detection_functions.convert_gray_scale(white_yellow)
        smooth_gray  = lane_detection_functions.apply_smoothing(gray)
        edges        = lane_detection_functions.detect_edges(smooth_gray)
        regions      = lane_detection_functions.select_region(edges)
        lines        = lane_detection_functions.hough_lines(regions)
        left_line, right_line = lane_detection_functions.lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line

        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lane_lines(image, (left_line, right_line))