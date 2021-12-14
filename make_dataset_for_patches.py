"""
Loads the nifti roi data of all mice and save the individual (globally standardised) slices to images in an ImageNet directory structure
"""


#%%
import ipdb
import numpy as np
import nibabel as nib
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

#%%
def load_all_rois(dir):
    """
    loads all the nifti rois of the mouse folder given in dir
    :param dir: the folder path to the nifti rois of the mouse
    :return: dict of rois nib object, dict of rois numpy array data
    """
    rois_nib = {}
    rois_data = {}
    for roi in dir.glob("*.nii.gz"):
        key = f'r{roi.name[1]}'
        rois_nib[key] = nib.load(roi)
        rois_data[key] = np.asarray(rois_nib[key].get_fdata())[:20,:32,:20,:3] # keeping the dimensions fixed (20, 32, 20, 3)

    return rois_nib, rois_data


#%%

def reshaped(X):
    return X.swapaxes(0, 1).reshape([32, -1])


def min_max_global_scaler(all_roi_data_list, keys, feature_range=(0,1)):
    X = np.concatenate([reshaped(data[:, :, :, 2]) for data in all_roi_data_list], axis=0)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_slices = scaler.fit_transform(X)

    global_scaled_slices = {}
    for idx, k in enumerate(keys):
        global_scaled_slices[k] = scaled_slices[idx * 32: idx * 32 + 32].reshape(-1,20,20)

    return global_scaled_slices


#%%
@dataclass
class Region():
    mouse: str
    region: str
    fdg_uptake: int
    fdg_class: str  # 0-5: 0, 5-10:1, 10-15:2, 15-20:3
    weight: float = 0
    adjusted_weight: float = 0
    meoh: float = 0
    protein: float = 0



def get_ground_truth_labels():
    """
    Returns FDG uptake class labels
    :return:
    """
    omics_file_path = Path(os.getenv("OMICS_DATA")) / "ERC15_Metabolomi_data_norm.csv"
    metabolomi_data = pd.read_csv(omics_file_path, header=None)

    def set_row_as_column_header(df, row):
        df.columns = df.iloc[row]
        df.drop(df.index[row], inplace=True)
        df.set_index(df.columns[0], inplace=True)
        return df

    part1 = metabolomi_data[:7]
    part1 = set_row_as_column_header(part1, 0)
    mouse_regions = {}

    def get_region_info(row):
        global i
        mapping_dict = {'0--5': 0, '5--10': 1, '10--15': 2, '15--20': 3}
        if not np.isnan(float(row[0])):
            mouse_id = str(row[0]).zfill(3)
            region_id = "r" + str(row[1])[1:]
            if mouse_id not in mouse_regions:
                mouse_regions[mouse_id] = {}
            mouse_regions[mouse_id][region_id] = Region(mouse=mouse_id,
                                                        region=region_id,
                                                        fdg_class=mapping_dict[row[4]],
                                                        fdg_uptake=int(row[3]))

    part1.apply(lambda x: get_region_info(x), axis=0)

    return mouse_regions



def get_extra_information(mouse_regions):
    """
    Gets extra information like mouse weight, injected dose, etc. for each mouse for each region
    :return:
    """
    omics_file_path = Path(os.getenv("OMICS_DATA")) / "Metadata.csv"
    metadata = pd.read_csv(omics_file_path, sep=';')

    def get_float_value(value):
        if type(value) == str:
            return float(value.replace(',','.'))
        return value

    def get_metadata(row):
        name = row['short Name'].split('/')
        mouse_id = str(name[0][1:]).zfill(3)
        region_id = 'r' + str(name[1][1:])

        weight = get_float_value(row['Weight [mg]'])
        adjusted_weight = get_float_value(row['Adjusted Weights Weight [mg] PROTEOMIC Center'])
        meoh = get_float_value(row['… µl 85 % MeOH       (6µl MeOH/mg tissue)'])
        protein = get_float_value(row['Protein [µg/µl]'])

        mouse_regions[mouse_id][region_id].weight = weight
        mouse_regions[mouse_id][region_id].adjusted_weight = adjusted_weight
        mouse_regions[mouse_id][region_id].meoh = meoh
        mouse_regions[mouse_id][region_id].protein = protein

    metadata.apply(lambda x: get_metadata(x), axis=1)

    return mouse_regions




#%%
def save_slice_image_to_class_folder(standardised_rois, mouse_id):
    save_dir = Path(os.getenv("ERC_15_DATA_PATH")).parent / "processed" / "slices" / "images"

    data_files_for_mouse_id_by_class[mouse_id] = {"0": [],
                                                  "1": [],
                                                  "2": [],
                                                  "3": []
                                                  }

    data_files_for_mouse_id_by_region[mouse_id] = {k: [] for k in ground_truth_data[mouse_id].keys()}

    for roi, data in standardised_rois.items():
        if roi in missing_mouse_rois[mouse_id]:
            continue
        for slice in data:
            img = Image.fromarray(slice.astype(np.uint8))
            label = ground_truth_data[mouse_id][roi].fdg_class
            save_path = save_dir / str(label)
            save_path.mkdir(parents=True, exist_ok=True)
            total_existing_items = len(list(save_path.glob('*.png')))
            img.save(f"{save_path}/{total_existing_items + 1}.png")
            data_files_for_mouse_id_by_class[mouse_id][str(label)].append(f'{total_existing_items + 1}.png')
            data_files_for_mouse_id_by_region[mouse_id][roi].append(f'{total_existing_items + 1}.png')


#%%
## Print missing mouse
def get_missing_mice():
    a = list(ground_truth_data.keys())
    b = [i.name for i in list(roi_data_path.glob('*'))]
    missing_mice = [i for i in b if i not in a]
    print(missing_mice)
    return missing_mice


# %%
## Print missing rois for existing mouse
def get_missing_rois_for_existing_mice():
    a = {m_id: list(m.keys()) for m_id, m in ground_truth_data.items()}
    b = {}
    for mouse in roi_data_path.glob("*"):
        if os.path.isdir(mouse):
            b[mouse.name] = [f'r{i.name[1]}' for i in list(mouse.glob("*"))]

    missing_mouse_rois = {}
    for mouse, data in b.items():
        if mouse in missing_mice:
            continue
        missing_rois = [i for i in data if i not in list(ground_truth_data[mouse].keys())]
        missing_mouse_rois[mouse] = missing_rois
        if len(missing_rois) > 0:
            print(f"{mouse} --> {missing_rois}")


    return missing_mouse_rois



def write_samples_to_disk(samples, skipped_samples, skip_mouse_id, labels_parent_folder="labels", task_name="fdg_uptake_class"):
    # shuffle and split the dataset
    samples = np.array(samples)
    np.random.seed(0)
    np.random.shuffle(samples)

    total_items = samples.shape[0]

    train_test_ration = 0.8

    if skip_mouse_id == -1:
        labels_dir_name = "all_mouse_data"
    else:
        labels_dir_name = f"without_mouse_{skip_mouse_id}"

    labels_dir = Path(os.getenv('ERC_15_DATA_PATH')).parent / "processed" / labels_parent_folder / labels_dir_name
    labels_dir.mkdir(parents=True, exist_ok=True)

    # training data
    with open(labels_dir / "train.txt", 'w') as f:
        f.writelines("%s\n" % s for s in samples[:int(total_items * train_test_ration)])

    # test data
    with open(labels_dir / "test.txt", 'w') as f:
        if type(samples) == np.ndarray:
            samples = list(samples)
        if type(skipped_samples) == np.ndarray:
            skipped_samples = list(skipped_samples)
        samples += skipped_samples
        f.writelines("%s\n" % s for s in samples[int(total_items * train_test_ration):])

    print(f"Task: {task_name}, labels file created!")


def create_labels_file(skip_mouse_id=-1):
    """
    Making .txt file with training and test set samples for FDG uptake class.
    """
    samples = []
    skipped_samples = []
    slice_data_path = Path(os.getenv('ERC_15_DATA_PATH')).parent / "processed" / "slices" / "images"
    for class_dir in slice_data_path.glob("*"):
        if os.path.isdir(class_dir):
            for file in class_dir.glob("*.png"):
                if skip_mouse_id != -1 and \
                        file.name in data_files_for_mouse_id_by_class[skip_mouse_id][str(class_dir.name)]:
                    skipped_samples.append(f'{class_dir.name}/{file.name}')
                else:
                    samples.append(f'{class_dir.name}/{file.name}')

    write_samples_to_disk(samples, skipped_samples, skip_mouse_id)



def create_meta_labels(skip_mouse_id=-1):
    """
    Takes ground_truth_labels containing metadata for each mouse for each region and data_files_for_mouse_id information
    and creates train and test .txt files
    :return:
    """

    def save_for_task(task_name):
        samples = []
        skipped_samples = []
        for mouse_id, region_dict in data_files_for_mouse_id_by_region.items():
            for region_id, files in region_dict.items():
                for file in files:
                    classname = ground_truth_data[mouse_id][region_id].fdg_class
                    name = f'{classname}/{file}'
                    value = ground_truth_data[mouse_id][region_id].__dict__[task_name]
                    if not np.isnan(value):
                        if skip_mouse_id == mouse_id:
                            skipped_samples.append(f'{name} {value}')
                        else:
                            samples.append(f'{name} {value}')

        write_samples_to_disk(samples, skipped_samples, skip_mouse_id, f"metadata/{task_name}", task_name)


    tasks = ['weight', 'adjusted_weight', 'meoh', 'protein']
    for task in tasks:
        save_for_task(task)



def create_all_mouse_data(roi_data_path, missing_mice):
    for mouse in roi_data_path.glob("*"):
        if os.path.isdir(mouse) and mouse.name not in missing_mice:
            print(mouse.name)
            _, all_rois = load_all_rois(mouse)
            standardised_rois = min_max_global_scaler([all_rois[k] for k in sorted(list(all_rois.keys()))],
                                                      keys=sorted(list(all_rois.keys())),
                                                      feature_range=(0, 255.0))

            save_slice_image_to_class_folder(standardised_rois, mouse.name)
            print(f"Done for {mouse.name}")
            del standardised_rois
            del all_rois

    create_labels_file()
    create_meta_labels()


def create_leave_one_mouse_data(roi_data_path, missing_mice):
    """
    Create dataset recursively by leaving data from one mouse sample
    :param roi_data_path:
    :param missing_mice:
    :return:
    """

    all_mouse = [i.name for i in roi_data_path.glob("*") if os.path.isdir(i) and i.name not in missing_mice]

    # creating dataset recursively
    for skip_mouse in all_mouse:
        print("Skipping mouse", skip_mouse)
        for mouse in roi_data_path.glob("*"):
            if os.path.isdir(mouse) and mouse.name not in (missing_mice + [skip_mouse]):
                print(mouse.name)
                _, all_rois = load_all_rois(mouse)
                standardised_rois = min_max_global_scaler([all_rois[k] for k in sorted(list(all_rois.keys()))],
                                                          keys=sorted(list(all_rois.keys())),
                                                          feature_range=(0, 255.0))

                # save_slice_image_to_class_folder(standardised_rois, mouse.name,
                #                                  data_parent_folder="leave_one_mouse_data",
                #                                  data_folder=f"without_mouse_{skip_mouse}")
                print(f"Done for {mouse.name}")
                del standardised_rois
                del all_rois

        create_labels_file(skip_mouse_id=skip_mouse)
        create_meta_labels(skip_mouse_id=skip_mouse)

#%%

if __name__ == "__main__":
    roi_data_path = Path(os.getenv('ERC_15_DATA_PATH')) / "ERC15_nifti_rois"
    ground_truth_data = get_ground_truth_labels()  # fdg class labels
    ground_truth_data = get_extra_information(ground_truth_data)  # metadata
    missing_mice = get_missing_mice()
    missing_mouse_rois = get_missing_rois_for_existing_mice()
    data_files_for_mouse_id_by_class = {}
    data_files_for_mouse_id_by_region = {}

    create_all_mouse_data(roi_data_path, missing_mice)

    create_leave_one_mouse_data(roi_data_path, missing_mice)



#%%
"""
Problems:-

1. we don't have the fdg uptake data (metabolomics data) for a few mice 
    ['028', '048', '030', '047', '050', '051', '052', '053', '054']
    
2. for certain existing mice, for certain regions, we have the imaging data but no metabolomics data.  
    007 --> ['r2']
    012 --> ['r2', 'r3']
    021 --> ['r1', 'r5']
    024 --> ['r1', 'r2', 'r5', 'r6', 'r7']
    009 --> ['r2', 'r5']
    010 --> ['r4']
    011 --> ['r3', 'r6']
    014 --> ['r1', 'r2', 'r6', 'r7', 'r8']
    025 --> ['r4']

"""


############################################
#%%


