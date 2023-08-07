import numpy as np
import os
import urllib.request
import random
import pickle
from torch.utils.data import Dataset

class graspDataset(Dataset):
    def __init__(self, main_dir, object_dir, mode = 'train', split = {'train': 0.6, 'val': 0.2, 'test': 0.2}, limit_files = None, normalization = None, transform_joint = None, transform_object = None):
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0

        self.main_dir = main_dir
        self.object_dir = object_dir
        self.split = split
        self.limit_files = limit_files
        self.transform_joint = transform_joint
        self.transform_object = transform_object
        self.normalization = normalization
        if self.normalization is not None:
            self.mean = normalization[0]
            self.std = normalization[1]
            self.max = normalization[2]
            self.min = normalization[3]


        self.classes, self.class_to_idx = self._find_classes(self.main_dir)
        
        self.data_joints, self.labels, self.object_meshes, self.object_names = self.make_dataset(
            directory=self.main_dir,
            dir_object = self.object_dir,
            class_to_idx=self.class_to_idx,
            mode = mode,
        )

        self.cache = {}
        #self.min_value, self.max_value = self._find_min_max_values()
        

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def select_split(self, data_joints, labels, object_meshes, object_names, mode):
        """
        Depending on the mode of the dataset, deterministically split it.
        
        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image
        
        :returns (images, labels), where only the indices for the
            corresponding data split are selected.
        """
        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(data_joints)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)
        
        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)
        
        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train+num_valid]
        elif mode == 'test':
            idx = rand_perm[num_train+num_valid:]

        if self.limit_files:
            idx = idx[:self.limit_files]
        
        ## changed and not sure at all
        if isinstance(data_joints, list): 
            return list(np.array(data_joints)[idx]), list(np.array(labels)[idx]), list(np.array(object_meshes)[idx]), list(np.array(object_names)[idx])
        else: 
            return data_joints[idx], list(np.array(labels)[idx]), object_meshes[idx] ##################### esto no lo entiendo

    
    def make_dataset(self, directory, dir_object, class_to_idx, mode):
        """
        Create the image dataset by preparaing a list of samples
        Images are sorted in an ascending order by class and file name
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset, NOT the actual images
            - labels is a list containing one label per image
        """
        data_joints, labels, object_meshes, object_names = [], [], [], []

        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if fname.endswith(".npy"):
                        grasp_path = os.path.join(root, fname)

                        ### added/modified ###
                        # crop the object name from fname
                        object_file = os.path.splitext(fname)[0]
                        object_name = object_file.split('_')[:-2] 
                        object_name = '_'.join(object_name) + '.npy'
                        object_path = os.path.join(dir_object, object_name)
                        ################

                        data_joints.append(grasp_path)
                        labels.append(label)
                        object_meshes.append(object_path)
                        object_names.append(object_name)

        data_joints, labels, object_meshes, object_names = self.select_split(data_joints, labels, object_meshes, object_names, mode)

        assert len(data_joints) == len(labels)
        assert len(data_joints) == len(object_meshes)

        return data_joints, labels, object_meshes, object_names
    
    
    def _normalize_data(self, data):

        normalized_data = (data - self.min) / ((self.max - self.min) + 1e-12)  # Normalize between 0 and 1
        normalized_data = normalized_data * 2 - 1  # Normalize between -1 and 1
        return normalized_data
    
    
    def _distribute_data(self, data):

        normalized_mean = (self.mean - self.min) / (self.max - self.min)
        normalized_mean = normalized_mean * 2 - 1

        normalized_std = (self.std - self.min) / (self.max - self.min)
        normalized_std = normalized_std * 2 - 1

        distributed_data = (data - normalized_mean) / (normalized_std)  # Normalize
        return distributed_data


    def __len__(self):
        
        return len(self.data_joints)
        #return 10000
                
    def __getitem__(self, index):

        '''
        if index >= 10000:
            raise StopIteration

        

        if index >= 5000:
            index = 1

        else:
            index = 0
        '''

        if index not in self.cache:

            data_dict = np.load(self.data_joints[index], allow_pickle=True).item() #loaded as np.array, need to add .item() to convert it to dictionary type

            data_joint = data_dict['grasp']
            #if self.transform_joint is not None:
            #    data_joint = self.transform_joint(data_joint)

            # Normalize and redistribute joint data
            if self.normalization is not None:
                data_joint = self._normalize_data(data_joint)
                #data_joint = self._distribute_data(data_joint)

            scale = data_dict['scale']

            data_obj = np.load(self.object_meshes[index], allow_pickle=True) * scale
            #if self.transform_object is not None:
            #    data_obj = self.transform_object(data_obj)

            label = int(self.labels[index])

            # Create a one-hot vector
            one_hot_vector = np.zeros(len(self.classes))
            one_hot_vector[label] = 1

            # Object names
            name = self.object_names[index]

            data_dict = {"joints":data_joint,"label":one_hot_vector, "mesh": data_obj, 'obj_name': name}
            data_dict = [data_joint, one_hot_vector, data_obj, name] # Make it like a pytorch ImageFolderDataset output
            
            self.cache[index] = data_dict

        return self.cache[index]