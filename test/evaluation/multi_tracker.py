import importlib
import os
import cv2 as cv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np



class Tracker:

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    import cv2 as cv

    def run_image_sequence(self, image_files, optional_box=None, debug=None, save_results=False):
        """Run the tracker on a list of image files."""
        params = self.get_parameters()
        params.debug = debug if debug is not None else getattr(params, 'debug', 0)

        tracker = self.create_tracker(params)
        params.tracker_name = self.name
        params.param_name = self.parameter_name

        # Check if the image files list is empty
        if not image_files:
            print("No images provided.")
            return

        output_boxes = []
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)

        # Initialize tracker on the first image
        first_image = self._read_image(image_files[0])

        # Convert BGR to RGB
        first_image = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)

        if optional_box is not None:
            tracker.initialize(first_image, {'init_bbox': optional_box})
            output_boxes.append(optional_box)
        else:
            frame_disp = first_image.copy()
            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                       (0, 0, 0), 1)
            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker.initialize(first_image, {'init_bbox': init_state})
            output_boxes.append(init_state)

        # Track the object in each subsequent frame
        for image_file in image_files[1:]:
            frame = self._read_image(image_file)

            # Convert BGR to RGB
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame_disp = frame.copy()

            # Draw the box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            cv.rectangle(frame_disp, (state[0], state[1]), (state[0] + state[2], state[1] + state[3]), (0, 255, 0), 5)

            # Display frame
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break

        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            results_path = os.path.join(self.results_dir, 'image_sequence_results.txt')
            np.savetxt(results_path, np.array(output_boxes), delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



