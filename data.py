import os
import numpy as np
import json
import cv2
from numpy.core.fromnumeric import resize


class PIEDataSet:
    num_subject = 25
    ratio_train = 0.7
    num_all_subject = 68

    def __init__(self, data_root, meta_file):
        self.data_root = data_root
        self.meta_file = meta_file

        if os.path.exists(self.meta_file):
            print(f'Meta file exists and load from {self.meta_file}')
            self.subject, self.train_list, self.test_list = self.load_meta()
        else:
            print(f'{self.meta_file} not found')
            self.subject = self.select_subject()
            self.train_list, self.test_list = self.train_test_split()
            print(f'Meta file saved to {self.meta_file}')
            self.save_meta()

        print(f'Num subject {len(self.subject)}, Num train {len(self.train_list)}, Num Test {len(self.test_list)}')
        self.label2gt = {c: i for i, c in enumerate(self.subject)}

    def select_subject(self):
        print('Selecting Subjects')
        all_subject = [str(s+1) for s in range(self.num_all_subject)]
        np.random.shuffle(all_subject)
        subject = all_subject[:self.num_subject]
        subject.append('selfie_32x32')
        return subject

    def train_test_split(self):
        print('Spliting Training and Testing')
        train_list = []
        test_list = []
        for s in self.subject:
            subject_path = os.path.join(self.data_root, s)
            all_samples = os.listdir(subject_path)
            train_num = int(len(all_samples) * self.ratio_train)
            np.random.shuffle(all_samples)
            train_list.extend([(os.path.join(subject_path, sample), s)
                               for sample in all_samples[:train_num]])
            test_list.extend([(os.path.join(subject_path, sample), s)
                              for sample in all_samples[train_num:]])

        return train_list, test_list

    def save_meta(self):
        with open(self.meta_file, 'w') as meta:
            data = dict(train_list=self.train_list,
                        test_list=self.test_list,
                        subject=self.subject)
            json.dump(data, meta, indent=1)

    def load_meta(self):
        with open(self.meta_file, 'r') as meta:
            data = json.load(meta)
        return data['subject'], data['train_list'], data['test_list']

    def load_data(self, train=True):
        imgs = []
        gts = []
        if train:
            datalist = self.train_list
        else:
            datalist = self.test_list
        for path, label in datalist:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            gt = self.label2gt[label]
            imgs.append(img)
            gts.append(gt)
        
        return np.stack(imgs)/255, np.stack(gts)


if __name__ == '__main__':
    PIEDataSet('./PIE/', './PIE/meta.json')
