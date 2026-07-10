from unimodal_pair_encoders.hague import config
import os
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tqdm
import pickle as pkl

def generate_png_figs_wrapper(object_dict, existing_images_dict, index_figs_path, cands_figs_path):
    new_generated_objects = {'cands': [], 'index': []}
    for file_type, figs_path in zip(['cands', 'index'], [cands_figs_path, index_figs_path]):
        for obj_ind, obj_data in tqdm.tqdm(object_dict[file_type].items()):
            if obj_ind in existing_images_dict[file_type]:
                continue
            generate_png_fig(obj_ind, obj_data['polygon_mesh'], figs_path)
            new_generated_objects[file_type].append(obj_ind)
        print(f"Generated {len(new_generated_objects[file_type])} {file_type} images")
    return new_generated_objects


def generate_figs_dir(dataset_name):
    code_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(code_dir, '..', '..', 'data', 'raw', 'hague'))
    figs_root = os.path.join(data_dir, 'png_figs', dataset_name)
    index_path = os.path.join(figs_root, 'index')
    cands_path = os.path.join(figs_root, 'cands')
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(cands_path, exist_ok=True)
    return index_path, cands_path


def get_existing_images_dict(index_path, cands_path):
    existing_images_dict = {}
    for file_type, dir_path in zip(['index', 'cands'], [index_path, cands_path]):
        existing_images_dict[file_type] = {f.split('.')[0] for f in os.listdir(dir_path) if f.endswith('.png')}
    return existing_images_dict


def find_min_coord(dim, polygon_mesh):
    return min([vertex[dim] for surface in polygon_mesh for vertex in surface])


def find_max_coord(dim, polygon_mesh):
    return max([vertex[dim] for surface in polygon_mesh for vertex in surface])


def generate_png_fig(obj_id, polygon_mesh, save_dir, margin=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for surface in polygon_mesh:
        poly = Poly3DCollection([surface], alpha=0.5, edgecolor='k')
        ax.add_collection3d(poly)

    x_min = find_min_coord(0, polygon_mesh) - margin
    y_min = find_min_coord(1, polygon_mesh) - margin
    z_min = find_min_coord(2, polygon_mesh) - margin
    x_max = find_max_coord(0, polygon_mesh) + margin
    y_max = find_max_coord(1, polygon_mesh) + margin
    z_max = find_max_coord(2, polygon_mesh) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)  # Disable grid
    ax.set_axis_off()  # Remove the background planes
    ax.set_rasterized(True)
    plt.savefig(os.path.join(save_dir, f'{obj_id}.png'), format='png', bbox_inches='tight',
                pad_inches=0, transparent=True)
    plt.close(fig)


def load_object_dict(logger, path, object_name='object_dict'):
    logger.info(f"Loading object_dict from {path}")
    try:
        object_dict = joblib.load(path)
        logger.info(f"{object_name} was loaded successfully")
        return object_dict
    except Exception as e:
        logger.error(f"Error happened while loading {object_name}: {e}. Starting building object_dict and "
                     f"prep_object_dict...")
        return None


def get_feature_name_list():
    return [f'{feature}_ratio' for feature in config.Features.object_properties]


def load_dataset_partition_dict(dataset_name, logger, seed):
    dir_path = config.FilePaths.dataset_partition_path
    logger.info(f"Loading dataset_partition_dict from {dir_path}{dataset_name}_seed{seed}.pkl")
    full_path = f"{dir_path}{dataset_name}_seed{seed}.pkl"
    dataset_partition_dict = pkl.load(open(full_path, 'rb'))
    logger.info(f"dataset_partition_dict was loaded successfully")
    return dataset_partition_dict


