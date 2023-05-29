# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# create_master.py
# 2023/05/29 to-arai antillia.com

import os
import cv2
import numpy as np
import glob
import shutil
import traceback
SIZE = 224

def create_master(input_dir, output_dir):
  test_train = ["test", "train"]
  for target in test_train:
    input_subdir = os.path.join(input_dir, target)
    output_subdir   = os.path.join(output_dir, target)
    if not os.path.exists(output_subdir):
      os.makedirs(output_subdir)
    categories = os.listdir(input_subdir)
    ANGLES = [0, 90, 180, 270]
    for category in categories:
      input_subdir_category = os.path.join(input_subdir, category)
      
      input_categorized_files = glob.glob(input_subdir_category + "/*.jpg")
      output_subdir_category = os.path.join(output_subdir, category)
      if not os.path.exists(output_subdir_category):
        os.makedirs(output_subdir_category)
      for input_categorized_file in input_categorized_files:
        if not input_categorized_file.endswith("-d.jpg"):
          image   = cv2.imread(input_categorized_file, cv2.IMREAD_COLOR)
          basename = os.path.basename(input_categorized_file)
          image   = cv2.resize(image, (SIZE, SIZE))
          if target == "train":
            for angle in ANGLES:
              rotated = rotate_one(image, angle,)
              name = "rotated_" + str(angle) + "_" + basename
              output_filepath = os.path.join(output_subdir_category, name)
              cv2.imwrite(output_filepath, rotated)
              print("--- Saved rotated image {}".format(output_filepath))
          else:
            shutil.copy2(input_categorized_file, output_subdir_category)
            print("=== Copied {} to {}".format(input_categorized_file, output_subdir_category))

def rotate_one(image, angle,):
    image = image.copy()

    h, w, _ = image.shape

    (cX, cY) = (w // 2, h // 2)
    MATRIX = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(MATRIX[0, 0])
    sin = np.abs(MATRIX[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    MATRIX[0, 2] += (nW / 2) - cX
    MATRIX[1, 2] += (nH / 2) - cY

    transformed = cv2.warpAffine(image, MATRIX, (nW, nH))
    return transformed


if __name__ == "__main__":
  try:
    input_dir  = "./Pap_Smear_Images"
    output_dir = "./PapSmearImages"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    create_master(input_dir, output_dir)

  except:
    traceback.print_exc()
