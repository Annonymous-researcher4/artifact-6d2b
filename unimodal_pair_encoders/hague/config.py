import os

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(os.path.join(_CODE_DIR, '..', '..', 'data', 'raw', 'hague'))


class FilePaths:
    object_dict_path = os.path.join(_DATA_DIR, 'object_dicts') + os.sep
    dataset_partition_path = os.path.join(_DATA_DIR, 'dataset_partitions') + os.sep


class Features:
    normalization = 'log_transform'  # geometric-property normalization
    max_ratio_val = 1000  # clip property ratios to avoid infinities
    object_properties = [
        "bounding_box_width", "bounding_box_length", "area", "perimeter", "perimeter_ind",
        "volume", "convex_hull_area", "convex_hull_volume", "ave_centroid_distance", "height_diff",
        "num_floors", "axes_symmetry", "compactness_2d", "compactness_3d", "density",
        "elongation", "shape_ind", "hemisphericality", "fractality", "cubeness", "circumference",
        "aligned_bounding_box_width", "aligned_bounding_box_length", "aligned_bounding_box_height",
        "num_vertices",
    ]



