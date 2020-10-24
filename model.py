from mrcnn.model import MaskRCNN
import numpy as np
import mrcnn.config
import mrcnn.utils
import cv2


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


class Model:
    def __init__(self, model_dir, weights_path):
        self.model = MaskRCNN(mode="inference", model_dir=model_dir, config=MaskRCNNConfig())
        self.model.load_weights(weights_path, by_name=True)

    @staticmethod
    def get_car_boxes(boxes, class_ids):
        car_boxes = []
        for i, box in enumerate(boxes):
            if class_ids[i] in [3]:
                car_boxes.append(box)
        return np.array(car_boxes)

    def get_occupancy_status(self, frame, slots):
        status = []
        frame = np.copy(frame)
        results = self.model.detect([frame], verbose=0)
        r = results[0]
        parked_car_boxes = self.get_car_boxes(r['rois'], r['class_ids'])

        overlaps = mrcnn.utils.compute_overlaps(slots, parked_car_boxes)
        n_cars = 0
        id=0
        for parking_area, overlap_areas in zip(slots, overlaps):
            max_iou_overlap = np.max(overlap_areas)
            y1, x1, y2, x2 = parking_area

            if max_iou_overlap < 0.25:
                status.append(False)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                status.append(True)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                n_cars += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(id), (x1 + 6, y1 - 6), font, 0.3, (255, 255, 0))
            id += 1
        return n_cars, status, frame
