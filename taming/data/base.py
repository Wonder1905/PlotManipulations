import bisect
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torch
import random
from itertools import permutations,product
class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImageTextPaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, augment=False,images_list_file=None,split="train"):
        self.size = size
        self.random_crop = random_crop
        self.augment = augment
        self.labels = dict() if labels is None else labels
        new_labels = {}
        if split=="train":
            self.data = torch.load("/workspace/FigureEdit/PlotsLegend/samples_dict.pth")
        else:
            self.data = torch.load("/workspace/FigureEdit/PlotsLegend/samples_dict_test.pth")
        new_data = {}
        hierercies = images_list_file.split("/")[:-1]
        data_path = "/".join(hierercies)
        for k,v in self.data.items():
            new_data[k] = {}
            for path,d  in v.items():
                if "/train" in path:
                    new_k1 = path.split("/train")[1]
                    connector = "/train/"
                elif "/test" in path:
                    new_k1 = path.split("/test")[1]
                    connector = "/test/"
                new_k1 = new_k1.replace("//", "/")#.split("/")[1]
                new_path = f"{data_path}/{connector}/{new_k1}"
                new_path = new_path.replace("///","/").replace("//","/")
                new_data[k][new_path] = d
        self.data = new_data
        self.data_keys = list(self.data.keys())
        self.images_list_file=images_list_file
        if self.size is not None and self.size > 0:
            self.rescaler = A.Resize(self.size,self.size)#A.SmallestMaxSize(max_size = self.size)
            if not self.random_crop and False:
                self.cropper = A.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = A.RandomCrop(height=self.size,width=self.size)
            #self.preprocessor = A.Compose([self.rescaler, self.cropper])
            self.preprocessor = A.Compose([self.rescaler])
        else:
            self.preprocessor = lambda **kwargs: kwargs
        self.colors = {
            'b': 'blue',
            'g': 'green',
            'r': 'red',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'k': 'black',
            'w': 'white'
        }
        self.colors2idx = {
            'b': 0,
            'g': 1,
            'r': 2,
            'c': 3,
            'm': 4,
            'y': 5,
            'k': 6,
            'w': 7,
            'x': 8,#x means no color!! because thereis not graph
        }
        
        self.permutations = list(product(list(self.colors2idx.keys()), repeat=3))
        self.color_permutation2index = {}
        j = 0
        for perm in self.permutations:
            if perm[0]=='x' or perm[1]=='x':
                continue
            self.color_permutation2index[perm]=j
            j+=1

        for col,_ in self.colors2idx.items():
            perm = (col,'x','x')
            self.color_permutation2index[perm] = j
            j+=1



        self.legend_location = {
            0: 'best (Axes only)',
            1: 'upper right',
            2: 'upper left',
            3: 'lower left',
            4: 'lower right',
            5: 'right',
            6: 'center left',
            7: 'center right',
            8: 'lower center',
            9: 'upper center',
            10: 'center'
        }
        if self.augment:
            # Add data aug transformations
            self.data_augmentation = A.Compose([
                A.GaussianBlur(p=0.1),
                A.OneOf([
                    A.HueSaturationValue (p=0.3),
                    A.ToGray(p=0.3),
                    A.ChannelShuffle(p=0.3)
                ], p=0.3)
            ])

    def __len__(self):
        len_of_dataset = len(self.data)
        return len_of_dataset

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.augment:
            image = self.data_augmentation(image=image)['image']
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    def combine_prompts(self,prompt_list):
        prompt_str = ""
        for idx,prompt in enumerate(prompt_list):
            prompt_str+=prompt
            if random.random()>0.5:
                prompt_str += " and "
            else:
                prompt_str += ", "
            if idx==len(prompt_list)-1:
                continue
        return  prompt_str
    def check_difference(self,chosen_samples):
        prompt_list = []
        meta_data = {"input":{},"output":{}}
        color_list_in = []
        color_list_out = []
        for k,v in chosen_samples[0].items():
            if k=='legend_location_pixels':
                continue
            v1 = chosen_samples[1][k]
            if k=='lines':
                lop_concat_out=""
                lop_concat_in=""
                for line_idx,line in v.items():
                    # meta_data["input"][f"line{line_idx}"] = {}
                    # meta_data["output"][f"line{line_idx}"] = {}
                    for k_in_line,v_in_line in line.items():
                        v1_in_line = v1[line_idx][k_in_line]
                        if k_in_line=="legend_of_plot":
                            name_of_line = v_in_line
                            lop_concat_in = lop_concat_in + "<s>" + v_in_line + "</s>"
                            lop_concat_out = lop_concat_out + "<s>" + v1_in_line + "</s>"


                        if v_in_line!= v1_in_line:#if they are different we need to put it in the prompt, otherwise just save it in a metadata dict for auxilary losses
                            if k_in_line == "legend_of_plot":
                                if random.random() > 0.5:
                                    prompt_list.append(f"change '{name_of_line}' name to '{v1_in_line}'")
                                else:
                                    prompt_list.append(f"change '{name_of_line}' to '{v1_in_line}'")
                            elif k_in_line=="color":
                                if random.random()>0.5:
                                    prompt_list.append(f"change '{k_in_line}' color to {self.colors[v1_in_line]}")
                                else:
                                    prompt_list.append(f"change '{k_in_line}' to {self.colors[v1_in_line]}")
                                color_list_in.append(v_in_line)
                                color_list_out.append(v1_in_line)
                                #meta_data["input"][f"line{line_idx}"][k_in_line]  = self.colors2idx[v_in_line]
                                #meta_data["output"][f"line{line_idx}"][k_in_line] = self.colors2idx[v1_in_line]
                        else:
                            if k_in_line=="color": #lop is saved in lop concat
                                #meta_data["input"][f"line{line_idx}"][k_in_line] = meta_data["output"][f"line{line_idx}"][k_in_line] = self.colors2idx[v_in_line]
                                color_list_in.append(v_in_line)
                                color_list_out.append(v1_in_line)

                meta_data["input"]["lop_concat"]=lop_concat_in
                meta_data["output"]["lop_concat"]=lop_concat_out
            elif v!=v1:
                if k=="legend_location":
                    #v1 = self.legend_location[v1]
                    prompt_list.append(f"change {k.replace('_',' ')} to '{v1}'")
                    meta_data["input"]["legend_location"]  = v
                    meta_data["output"]["legend_location"] = v1
                elif k=="title":
                    meta_data["input"]["title"]  = v1
                    meta_data["output"]["title"] = v
                    if v1=="" and random.random()>0.5:
                        prompt_list.append(f"remove title")
                    else:
                        prompt_list.append(f"change {k} to '{v1}'")
            elif v==v1:
                if k=="title":
                    meta_data["input"]["title"] = meta_data["output"]["title"] = v
                elif k=="legend_location":
                    meta_data["input"]["legend_location"] = meta_data["output"]["legend_location"] = v
        for _ in range(3-len(color_list_in)):
            color_list_in.append('x')
        for _ in range(3-len(color_list_out)):
            color_list_out.append('x')
        meta_data["input"]["colors"]  = self.color_permutation2index[tuple(color_list_in)]
        meta_data["output"]["colors"] = self.color_permutation2index[tuple(color_list_out)]
        meta_data["input"]["legend_location_pixels"] = chosen_samples[0]["legend_location_pixels"]
        meta_data["output"]["legend_location_pixels"]= chosen_samples[1]["legend_location_pixels"]
        return prompt_list,meta_data
    def __getitem__(self, i):
        sample={}
        key = self.data_keys[i]
        conent_idx = self.data[key]
        value_list = list(conent_idx.values())
        key_list = list(conent_idx.keys())
        chosen_samples =  random.sample(list(np.arange(len(list(value_list)))), 2)
        value_chosen = np.array(value_list)[np.array(chosen_samples)]
        keys_chosen = np.array(key_list)[np.array(chosen_samples)]
        sample["input_image"] = self.preprocess_image(keys_chosen[0])
        sample["file_path_"]  =  keys_chosen[0]
        sample["output_image"] = self.preprocess_image(keys_chosen[1])
        prompt_list,meta_data = self.check_difference(value_chosen)
        sample["prompt"] = self.combine_prompts(prompt_list).replace("\\","")
        return sample,meta_data
class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None, augment=False,images_list_file=None):
        self.size = size
        self.random_crop = random_crop
        self.augment = augment
        self.labels = dict() if labels is None else labels
        new_labels = []
        hierercies = images_list_file.split("/")[:-1]
        data_path = "/".join(hierercies)
        for path in paths:
            if "/train" in path:
                new_k1 = path.split("/train")[1]
                connector = "/train/"
            elif "/test" in path:
                new_k1 = path.split("/test")[1]
                connector = "/test/"
            new_k1 = new_k1.replace("//", "/")#.split("/")[1]
            new_path = f"{data_path}/{connector}/{new_k1}"
            new_path = new_path.replace("///","/").replace("//","/")
            new_labels.append(new_path)
        self.labels["file_path_"] = new_labels
        self._length = len(new_labels)
        self.images_list_file=images_list_file
        if self.size is not None and self.size > 0:
            self.rescaler = A.Resize(self.size,self.size)#A.SmallestMaxSize(max_size = self.size)
            if not self.random_crop and False:
                self.cropper = A.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = A.RandomCrop(height=self.size,width=self.size)
            #self.preprocessor = A.Compose([self.rescaler, self.cropper])
            self.preprocessor = A.Compose([self.rescaler])
        else:
            self.preprocessor = lambda **kwargs: kwargs

        if self.augment:
            # Add data aug transformations
            self.data_augmentation = A.Compose([
                A.GaussianBlur(p=0.1),
                A.OneOf([
                    A.HueSaturationValue (p=0.3),
                    A.ToGray(p=0.3),
                    A.ChannelShuffle(p=0.3)
                ], p=0.3)
            ])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.augment:
            image = self.data_augmentation(image=image)['image']
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
