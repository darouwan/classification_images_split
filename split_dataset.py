import os

import cv2
import pandas as pd
import splitfolders

from data_augmentation import data_augmentation

# Split with a ratio.
# To only split into training and validation set,
# set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio('input',
                   output='output',
                   seed=1337,
                   ratio=(.8, .2),
                   group_prefix=None,
                   move=False)

data_argumentation_threshold = 0.8
data_argumentation_factor = 5

cls_count = []
for path, dirs, files in os.walk('output/train'):
    if len(files) == 0:
        continue
    cls = os.path.split(path)[-1]
    # print(cls, files)
    cls_count.append({'cls': cls, 'count': len(files)})

cls_count_df = pd.DataFrame(cls_count)

cls_count_df['minimum_count'] = cls_count_df['count'].max(
) * data_argumentation_threshold
print(cls_count_df)
data_argumentation_df = cls_count_df[cls_count_df['minimum_count'] >
                                     cls_count_df['count']]
print(f'data_argumentation_df = \n{data_argumentation_df}')
data_argumentation_cls_list = data_argumentation_df['cls'].tolist()
print(f'data_argumentation_cls_list={data_argumentation_cls_list}')

for cls in data_argumentation_cls_list:
    file_dir = os.path.join('output/train', cls)
    files = os.listdir(file_dir)
    for file in files:
        for i in range(data_argumentation_factor):
            img = cv2.imread(os.path.join(file_dir, file))
            img = data_augmentation(img)
            new_file_name = os.path.splitext(
                file)[0] + f'_{i}' + os.path.splitext(file)[1]
            cv2.imwrite(os.path.join(file_dir, new_file_name), img)
