import os
import numpy as np
from numpy import newaxis
from scipy import ndimage


position_i = ["justAPlaceholder", "symbol_1", "symbol_2",
              "symbol_3", "symbol_4", "symbol_5",
              "symbol_6", "symbol_7", "symbol_8",
              "symbol_9", "symbol_10", "symbol_11",
              "symbol_12", "symbol_13", "symbol_14",
              "symbol_15", "symbol_16", "symbol_17"]


def token_position(x):

    return int(x.split('_')[-1]) - 1


def subject_encode(x):

    return int(x[1:]) - 1

    
def export_data(data_dir='/content/drive/MyDrive/Inbed_Classification/dataset/experiment-i',
                preprocess=True):
    data_dict = dict()
    for _, dirs, _ in os.walk(data_dir):
        for directory in dirs:
            # each directory is a subject
            subject = directory
            data = None
            labels = None
            for _, _, files in os.walk(os.path.join(data_dir, directory)):
                for file in files:
                    file_path = os.path.join(data_dir, directory, file)
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                            raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            if preprocess:
                                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                                # Spatio-temporal median filter 3x3x3
                                raw_data = ndimage.median_filter(raw_data, 3)
                                past_image = ndimage.median_filter(past_image, 3)
                                future_image = ndimage.median_filter(future_image, 3)
                                raw_data = np.concatenate((raw_data[newaxis, :, :], past_image[newaxis, :, :], future_image[newaxis, :, :]), axis=0)
                                raw_data = np.median(raw_data, axis=0)
                            # Change the range from [0-1000] to [0-255].
                            file_data = np.round(raw_data * 255 / 1000).astype(np.uint8)
                            file_data = file_data.reshape(1, 64, 32)

                            # Turn the file index into position list,
                            # and turn position list into reduced indices.
                            file_label = token_position(position_i[int(file[:-4])])
                            file_label = np.array([file_label])

                            if data is None:
                                data = file_data
                            else:
                                data = np.concatenate((data, file_data), axis=0)
                            if labels is None:
                                labels = file_label
                            else:
                                labels = np.concatenate((labels, file_label), axis=0)

            data_dict[subject] = (data, labels)

    return data_dict


def data_split(subject_out):
    data = export_data()
    keys = list(data.keys())

    train_data = dict()
    train_data["images"] = None
    train_data["postures"] = None

    test_data = dict()

    train_keys = [key for key in keys if key != subject_out]

    print(train_keys)
    print(subject_out)

    for key in train_keys:
        if train_data["images"] is None:
            train_data["images"] = data[key][0]
            train_data["postures"] = data[key][1]
        else:
            train_data["images"] = np.concatenate((train_data["images"], data[key][0]), axis=0)
            train_data["postures"] = np.concatenate((train_data["postures"], data[key][1]), axis=0)

    test_data["images"] = data[subject_out][0]
    test_data["postures"] = data[subject_out][1]
    print(f"Train: {train_data['images'].shape[0]}, Test: {test_data['images'].shape[0]}")
    
    return train_data, test_data

