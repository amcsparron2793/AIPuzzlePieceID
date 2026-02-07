import json
from io import TextIOWrapper
from pathlib import Path
from typing import Optional


class _JsonLabelLoaderMixin:
    RESULT_KEYS = ['value', 'rectanglelabels']
    VALUE_KEYS = ['x', 'y', 'width', 'height']

    @staticmethod
    def _convert_to_decimal(percent: float) -> float:
        return percent / 100.0

    @classmethod
    def _validate_result_dict(cls, result: dict):
        if any([x for x in cls.RESULT_KEYS if x not in result]):
            raise ValueError(f"Invalid result format: {result}")

        if any([x for x in cls.VALUE_KEYS if x not in result["value"]]):
            raise ValueError(f"Invalid value format: {result['value']}")
        return result['value']

    def _get_bounding_box_info(self, value: dict):
        # Extract bounding box information
        x = self._convert_to_decimal(value["x"])  # Convert from percentage to [0,1]
        y = self._convert_to_decimal(value["y"])
        width = self._convert_to_decimal(value["width"])
        height = self._convert_to_decimal(value["height"])
        return x, y, width, height

    def _extract_json_info(self, result: dict):
        value = self._validate_result_dict(result)
        x, y, width, height = self._get_bounding_box_info(value)

        # Determine if it's an edge piece
        is_edge = 1.0 if "Edge" in value["rectanglelabels"][0] else 0.0
        # Assuming format: [x_center, y_center, width, height, confidence]
        final_values = [x, y, width, height, is_edge]
        return final_values


class _LabelLoaderMixin(_JsonLabelLoaderMixin):
    def __init__(self):
        self.detections = []
        self.label_dir: Optional[Path] = None

    def _load_from_plaintext(self, f:TextIOWrapper):
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            # Assuming format: [x_center, y_center, width, height, confidence]
            self.detections.append(values)

    def _load_from_json(self, f:TextIOWrapper):
        item = json.load(f)
        # Each image can have multiple annotation sets, but typically just one
        for annotation_set in item["annotations"]:
            # Each annotation set has multiple results (individual bounding boxes)
            for result in annotation_set["result"]:
                values = self._extract_json_info(result)
                self.detections.append(values)

    def load_labels(self, img_file:Path = None, **kwargs):
        # Load labels if they exist
        #base_name = os.path.splitext(img_file)[0]
        base_name = kwargs.get('base_name', img_file.stem)
        file_extension = kwargs.get('file_extension', '.txt')
        label_path = Path(self.label_dir,
                          f"{base_name}{file_extension}")

        if label_path.exists():
            with open(label_path, 'r') as f:
                if file_extension == '.json':
                    self._load_from_json(f)
                elif file_extension == '.txt':
                    self._load_from_plaintext(f)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")
        else:
            print(f"No labels found for {img_file}. Skipping...")
