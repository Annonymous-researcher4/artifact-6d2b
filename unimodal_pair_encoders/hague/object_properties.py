from scipy.spatial import ConvexHull
import numpy as np
from unimodal_pair_encoders.hague import config
from collections import defaultdict
import math
from multiprocessing import Pool, cpu_count
from time import time


class ObjectPropertiesProcessor:
    def __init__(self, object_dict, vector_normalization):
        self.object_dict = object_dict
        self.x_coordinates = self._get_specific_coordinates(0)
        self.y_coordinates = self._get_specific_coordinates(1)
        self.z_coordinates = self._get_specific_coordinates(2)
        self.vector_normalization = vector_normalization
        self.cores_to_use = min(cpu_count(), len(config.Features.object_properties))
        self.eigen_dict = self._create_eigen_dict()
        self.prop_names_dict = self._get_property_dict()
        self.property_dict_generation_time = self._create_prop_vals_dict()

    def _get_specific_coordinates(self, coord_index):
        specific_coord_dict = defaultdict(dict)
        for object_type, object_dict in self.object_dict.items():
            if object_type not in ['cands', 'index']:
                continue
            for obj_ind, data in object_dict.items():
                coords = np.unique(np.array([coord[coord_index] for surface in data['polygon_mesh']
                                             for coord in surface]), axis=0)
                specific_coord_dict[object_type][obj_ind] = coords
        return specific_coord_dict

    def _create_prop_vals_dict(self):
        start_time = time()
        properties = config.Features.object_properties
        self.prop_vals_dict = {prop: {obj_type: {} for obj_type in ['cands', 'index']} for prop in properties}
        args = [
            (prop, self.object_dict, self.prop_names_dict[prop], self.vector_normalization)
            for prop in properties
        ]
        with Pool(processes=self.cores_to_use - 2) as pool:
            aggregated_prop_dicts = pool.starmap(ObjectPropertiesProcessor.process_prop, args)

        for res in aggregated_prop_dicts:
            self.prop_vals_dict.update(res)
        end_time = time()
        return round(end_time - start_time, 2)

    @staticmethod
    def process_prop(prop, object_dict, prop_func, vector_normalization):
        curr_prop_dict = {prop: {}}
        for obj_type, objs in object_dict.items():
            if obj_type not in ['cands', 'index']:
                continue
            curr_prop_dict[prop][obj_type] = {}
            for obj_ind in objs.keys():
                val = prop_func(obj_type, obj_ind)
                if vector_normalization:
                    val = np.log1p(val)
                curr_prop_dict[prop][obj_type][obj_ind] = val
        return curr_prop_dict

    def _get_property_dict(self):
        return {prop: getattr(self, f'_get_{prop}') for prop in config.Features.object_properties}

    def _get_bounding_box_width(self, object_type, obj_ind):
        x_coords = self.x_coordinates[object_type][obj_ind]
        return max(x_coords) - min(x_coords)

    def _get_bounding_box_length(self, object_type, obj_ind):
        y_coords = self.y_coordinates[object_type][obj_ind]
        return max(y_coords) - min(y_coords)

    def _get_aligned_min_max_values(self, object_type, obj_ind):
        vertices = self.object_dict[object_type][obj_ind]['vertices']
        eigenvectors = self.eigen_dict[object_type][obj_ind]['eigenvectors']
        aligned_vertices = np.dot(vertices, eigenvectors)
        min_vals = np.min(aligned_vertices, axis=0)
        max_vals = np.max(aligned_vertices, axis=0)
        return min_vals, max_vals

    def _get_aligned_bounding_box_width(self, object_type, obj_ind):
        min_vals, max_vals = self._get_aligned_min_max_values(object_type, obj_ind)
        return max_vals[0] - min_vals[0]

    def _get_aligned_bounding_box_length(self, object_type, obj_ind):
        min_vals, max_vals = self._get_aligned_min_max_values(object_type, obj_ind)
        return max_vals[1] - min_vals[1]

    def _get_aligned_bounding_box_height(self, object_type, obj_ind):
        min_vals, max_vals = self._get_aligned_min_max_values(object_type, obj_ind)
        return max_vals[2] - min_vals[2]

    def _get_area(self, object_type, obj_ind):
        if obj_ind not in self.prop_vals_dict['area'][object_type].keys():
            polygon_mesh = self.object_dict[object_type][obj_ind]['polygon_mesh']
            area = self._compute_polygon_mesh_area(polygon_mesh)
            self.prop_vals_dict['area'][object_type][obj_ind] = area
        else:
            area = self.prop_vals_dict['area'][object_type][obj_ind]
        return max(area, 1)

    def _compute_polygon_mesh_area(self, polygon_mesh):
        """
        Compute the total surface area of a mesh polygon.

        Parameters:
        - mesh_polygon: List of surfaces, where each surface is a list of vertices (each vertex is a 3D point).

        Returns:
        - Total surface area of the mesh polygon.
        """
        total_area = 0.0
        for surface in polygon_mesh:
            total_area += self._compute_polygon_area(surface)
        return total_area

    def _compute_polygon_area(self, polygon):
        """
        Compute the area of a polygon by summing the areas of its triangles.

        Parameters:
        - polygon: List of vertices (each vertex is a 3D point).

        Returns:
        - Area of the polygon.
        """
        area = 0.0
        if len(polygon) < 3:
            return area
        for i in range(1, len(polygon) - 1):
            # Form a triangle with the first vertex and two consecutive vertices
            triangle = [polygon[0], polygon[i], polygon[i + 1]]
            normal = np.cross(np.array(triangle[1]) - np.array(triangle[0]),
                              np.array(triangle[2]) - np.array(triangle[0]))
            area += 0.5 * np.linalg.norm(normal)
        return area

    def _compute_specific_polygon_perimeter(self, object_type, file_ind, reference_point):
        """
        Compute the perimeter of the lowest polygon in the object.

        Parameters:
        - object_type: Type of the object (cands or index).
        - file_ind: Index of the object in the object dictionary.
        - reference_point: Reference point to compute the perimeter (min_z or max_z).

        Returns:
        - Perimeter of the lowest/highest polygon in the object.
        """
        perimeter = 0.0
        for polygon in self.object_dict[object_type][file_ind]['polygon_mesh']:
            if all(vertex[2] == reference_point for vertex in polygon):
                for i in range(len(polygon)):
                    perimeter += np.linalg.norm(np.array(polygon[i]) - np.array(polygon[(i + 1) % len(polygon)]))
                break
        return perimeter

    def _get_perimeter(self, object_type, file_ind):
        """
        Get the perimeter of the object.

        Parameters:
        - object_type: Type of the object (cands or index).
        - file_ind: Index of the object in the object dictionary.

        Returns:
        - Perimeter of the object.
        """
        if file_ind not in self.prop_vals_dict['perimeter'][object_type].keys():
            min_z = min(self.z_coordinates[object_type][file_ind])
            max_z = max(self.z_coordinates[object_type][file_ind])
            perimeter = self._compute_specific_polygon_perimeter(object_type, file_ind, min_z)
            if perimeter == 0.0:
                perimeter = self._compute_specific_polygon_perimeter(object_type, file_ind, max_z)
                perimeter = max(perimeter, 1)
            self.prop_vals_dict['perimeter'][object_type][file_ind] = perimeter
        else:
            perimeter = self.prop_vals_dict['perimeter'][object_type][file_ind]
        return perimeter

    def _get_perimeter_ind(self, object_type, obj_ind):
        r"""
        Get the perimeter index of the object, computed as $2 \sqrt{\pi \cdot Area} / Perimeter$.

        Parameters:
        - object_type: Type of the object (cands or index).
        - obj_ind: Index of the object in the object dictionary.

        Returns:
        - Perimeter index of the object
        """
        area = self._get_area(object_type, obj_ind)
        perimeter = self._get_perimeter(object_type, obj_ind)
        return 2 * math.sqrt(math.pi * area) / perimeter

    def _get_volume(self, object_type, obj_ind):
        if obj_ind not in self.prop_vals_dict['volume'][object_type].keys():
            polygon_mesh = self.object_dict[object_type][obj_ind]['polygon_mesh']
            volume = 0.0
            for polygon in polygon_mesh:
                for i in range(1, len(polygon) - 1):
                    triangle = [polygon[0], polygon[i], polygon[i + 1]]
                    volume += np.dot(triangle[0], np.cross(triangle[1], triangle[2])) / 6.0
            volume = abs(volume)
            self.prop_vals_dict['volume'][object_type][obj_ind] = volume
        else:
            volume = self.prop_vals_dict['volume'][object_type][obj_ind]
        return volume

    def _get_convex_hull_area(self, object_type, obj_ind):
        vertices = self.object_dict[object_type][obj_ind]['vertices']
        vertices_2d = np.array(vertices)[:, :2]
        return ConvexHull(vertices_2d).area

    def _get_convex_hull_volume(self, object_type, obj_ind):
        vertices = self.object_dict[object_type][obj_ind]['vertices']
        return ConvexHull(vertices).volume

    def _get_ave_centroid_distance(self, object_type, obj_ind):
        vertices = self.object_dict[object_type][obj_ind]['vertices']
        centroid = self.object_dict[object_type][obj_ind]['centroid']
        return np.mean([np.linalg.norm(np.array(vertex) - np.array(centroid)) for vertex in vertices])

    def _get_max_height(self, object_type, obj_ind):
        return max(self.z_coordinates[object_type][obj_ind])

    def _get_min_height(self, object_type, obj_ind):
        return min(self.z_coordinates[object_type][obj_ind])

    def _get_height_diff(self, object_type, obj_ind):
        return self._get_max_height(object_type, obj_ind) - \
               self._get_min_height(object_type, obj_ind)

    def _get_num_floors(self, object_type, obj_ind):
        return len(set(self.z_coordinates[object_type][obj_ind]))

    def _get_axes_symmetry(self, object_type, obj_ind):
        x_coords = self.x_coordinates[object_type][obj_ind]
        y_coords = self.y_coordinates[object_type][obj_ind]
        z_coords = self.z_coordinates[object_type][obj_ind]
        return np.mean([np.std(x_coords), np.std(y_coords), np.std(z_coords)])

    def _get_compactness_2d(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        convex_hull_area = self._get_convex_hull_area(object_type, obj_ind)
        return area / convex_hull_area

    def _get_compactness_3d(self, object_type, obj_ind):
        volume = self._get_volume(object_type, obj_ind)
        convex_hull_volume = self._get_convex_hull_volume(object_type, obj_ind)
        return volume / convex_hull_volume

    def _get_density(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        perimeter = self._get_perimeter(object_type, obj_ind)
        return area / perimeter

    def _create_eigen_dict(self):
        eigen_dict = dict()
        for obj_type, object_dict in self.object_dict.items():
            if obj_type not in ['cands', 'index']:
                continue
            eigen_dict[obj_type] = dict()
            for obj_ind, obj_data in object_dict.items():
                vertices = obj_data['vertices']
                eigen_dict[obj_type][obj_ind] = dict()
                covariance_matrix = np.cov(vertices, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                eigen_dict[obj_type][obj_ind]['eigenvalues'] = eigenvalues
                eigen_dict[obj_type][obj_ind]['eigenvectors'] = eigenvectors
        return eigen_dict

    def _get_elongation(self, object_type, obj_ind):
        eigenvalues = self.eigen_dict[object_type][obj_ind]['eigenvalues']
        return np.sqrt(eigenvalues.max() / eigenvalues.min())

    def _get_shape_ind(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        perimeter = self._get_perimeter(object_type, obj_ind)
        return perimeter / math.sqrt(4 * np.pi * area)

    def _get_hemisphericality(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        volume = self._get_volume(object_type, obj_ind)
        return 3 * math.sqrt(2) * math.sqrt(math.pi) * volume / (math.pow(area, 1.5))

    def _get_fractality(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        volume = self._get_volume(object_type, obj_ind)
        return 1 - math.log(volume) / (1.5 * math.log(area))

    def _get_cubeness(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        volume = self._get_volume(object_type, obj_ind)
        return 6 * math.pow(volume, 2/3) / area

    def _get_circumference(self, object_type, obj_ind):
        area = self._get_area(object_type, obj_ind)
        volume = self._get_volume(object_type, obj_ind)
        return 4 * math.pi * math.pow(3 * volume / (4 * math.pi), 2/3) / area

    def _get_num_vertices(self, object_type, obj_ind):
        return len(self.object_dict[object_type][obj_ind]['vertices'])
