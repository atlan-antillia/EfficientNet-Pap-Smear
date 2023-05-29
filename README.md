<h2>
EfficientNet Pap Smear (Updated: 2023/05/30)
</h2>
<a href="#1">1 EfficientNetV2 Pap Smear Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Install Python packages</a><br>
<a href="#2">2 Python classes for Pap Smear Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Pap Smear Classification</a>
</h2>

This is an experimental project Pap Smear Classification based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>
The Pap Smear dataset used here has been taken from the following web site:<br>
 <a href="http://mde-lab.aegean.gr/index.php/downloads">PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES</a>
<br>
<br>

 We use python 3.8 and tensorflow 2.10.1 environment on Windows 11 for this project.<br>
<br>
<li>
2023/05/30 Recreated Augmented <b>PapSmearImage/train</b> dataset by resizing and rotating the original images. 
</li>
<br>   
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Pap-Smear.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
├─efficientnetv2-b0
└─projects
    └─Pap-Smear
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        │  └─chief
        ├─PapSmearImages
        │  ├─test
        │  │  ├─carcinoma_in_situ
        │  │  ├─light_dysplastic
        │  │  ├─moderate_dysplastic
        │  │  ├─normal_columnar
        │  │  ├─normal_intermediate
        │  │  ├─normal_superficiel
        │  │  └─severe_dysplastic
        │  └─train
        │      ├─carcinoma_in_situ
        │      ├─light_dysplastic
        │      ├─moderate_dysplastic
        │      ├─normal_columnar
        │      ├─normal_intermediate
        │      ├─normal_superficiel
        │      └─severe_dysplastic
        └─test
</pre>

<h3>
<a id="1.2">1.2 Prepare Pap_Smear_Images</a>
</h3>
<h3>
1.2.1 Expand original images
</h3>

 We have created <b>Resized_Pap_Smear_Images_jpg_200x200_master</b> dataset 
 by using <a href="./projects/Pap-Smear/expand.py">expand.py</a> script from the original dataset
<b>New database pictures</b> downloaded from the following website:<br> 
 <a href="http://mde-lab.aegean.gr/index.php/downloads">PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES</a>
<br> 
<h3>
1.2.2 Split master
</h3>
 Furthermore, we have created <b>Pap_Smear_Image</b> dataset from <b>Resized_Pap_Smear_Images_jpg_200x200_master</b> 
 by using <a href="./projects/Pap-Smear/split_master.py">split_master.py</a> script.<br>
<br>
<h3>
1.2.3 Augment train dataset
</h3>
 We have created augmented <b>PapSmearImage</b> dataset from the <b>Pap_Smear_Image/train</b> dataset by resizing
image files in that folder to 224x224, and rotating those resized images by angle in angles =[0, 90, 180, 270].<br>
<br>
 
The number of images of train and test image dataset:<br>
<img src="./projects/Pap-Smear/_PapSmearImages_.png" width="820" height="auto">
<br>
<br>
1 Sample images in PapSmearImages/train/carcinoma_in_situ:<br>
<img src="./asset/train_carcinoma_in_situ.png"  width="820" height="auto"><br><br>

2 Sample images in PapSmearImages/train/light_dysplastic:<br>
<img src="./asset/train_light_dysplastic.png"  width="820" height="auto"><br><br>

3 Sample images in PapSmearImages/train/moderate_dysplastic:<br>
<img src="./asset/train_moderate_dysplastic.png"  width="820" height="auto"><br><br>

4 Sample images in PapSmearImages/train/normal_columnar:<br>
<img src="./asset/train_normal_columnar.png"  width="820" height="auto"><br><br>

5 Sample images in PapSmearImages/train/normal_intermediate:<br>
<img src="./asset/train_normal_intermediate.png"  width="820" height="auto"><br><br>

6 Sample images in PapSmearImages/train/normal_superficiel:<br>
<img src="./asset/train_normal_superficiel.png"  width="820" height="auto"><br><br>

7 Sample images in PapSmearImages/train/severe_dysplastic:<br>
<img src="./asset/train_severe_dysplastic.png"  width="820" height="auto"><br><br>


<h3>
<a id="1.3">1.3 Install Python packages</a>
</h3>

Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Pap Smear Classification</a>
</h2>
We have defined the following python classes to implement our Pap Smear Classification.<br>

<li>
<a href="./ClassificationReportWriter.py">ClassificationResportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>

<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>

<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./TestDataset.py">TestDataset</a>
</li>
<br>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-b0</b> to train Pap Smear Classification Model by using
 the dataset <b>./projects/Pap-Smear/PapSmearImages/train</b>.

<br> 
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0.tgz">efficientnetv2-b0.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-b0
└─projects
    └─Pap-Smear
</pre>

<h2>
<a id="4">4 Train</a>
</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Pap Smear efficientnetv2-b0 model by 
using  the dataset <b>./projects/Pap-Smear/Pap_Smear_Images/train</b> derived from
 <a href="http://mde-lab.aegean.gr/index.php/downloads">PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES</a>
<br>
<pre>
./1_train.bat
</pre>
<pre>
rem 2023/05/29 
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --model_name=efficientnetv2-b0  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-b0/model ^
  --optimizer=adam ^
  --image_size=224 ^
  --eval_image_size=224 ^
  --data_dir=./PapSmearImages/train ^
  --model_dir=./models ^
  --data_augmentation=True ^
  --valid_data_augmentation=False ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --num_epochs=80 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  

</pre>
,where data_generator.config is the following<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=False
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
horizontal_flip    = True
vertical_flip      = True
 
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.2, 3.0]

data_format        = "channels_last"
[validation]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=False
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.1, 2.0]
data_format        = "channels_last"
</pre>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Pap-Smear/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Pap-Smear/eval/train_losses.csv">train_losses</a> files
<br>
<h3>
<a id="4.2">Training result</a>
</h3>

Training console output:<br>

<br>
<img src="./asset/Pap_Smear_train_console_output_at_epoch_45_0529.png" width="840" height="auto"><br>

As shown above, please note that the <b>best_model.h5</b> has been saved at epoch 17.
<br>
<br>
Train_accuracies:<br>
<img src="./projects/Pap-Smear/eval/train_accuracies.png" width="740" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Pap-Smear/eval/train_losses.png" width="740" height="auto"><br>

<br>

<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the Pap Smear test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=224 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
, where label_map.txt is the following:<br>
<pre>
carcinoma_in_situ
light_dysplastic
moderate_dysplastic
normal_columnar
normal_intermediate
normal_superficiel
severe_dysplastic
</pre>
<br>


<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Pap-Smear/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Pap-Smear/PapSmearImages/test">PapSmearImages/test</a>.
Pap-Smear/test:<br>
<img src="./asset/PapSmeartest.png" width="820" height="auto">
<br><br>


<h3>
<a id="5.3">5.3 Inference result</a>
</h3>

This inference command will generate <a href="./projects/Pap-Smear/inference/inference.csv">inference result file</a>.
<br>
Inference console output:<br>
<img src="./asset/Pap_Smear_infer_console_output_at_epoch_45_0529.png" width="840" height="auto"><br>
<br>
Inference result:<br>
<img src="./asset/Pap_Smear_inference_result_at_epoch_45_0529.png" width="740" height="auto"><br>


<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Pap-Smear/PapSmearImage/test">Pap-Smear/PapSmearImage/test dataset</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --data_dir=./PapSmearImages/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --eval_image_size=224 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --debug=False 
</pre>


<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Pap-Smear/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Pap-Smear/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Pap_Smear_evaluate_console_output_at_epoch_45_0529.png" width="840" height="auto"><br>
<br>

Classification report:<br>
<img src="./asset/Pap_Smear_classification_report_at_epoch_45_0529.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Pap-Smear/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<h3>References</h3>
<b>1,PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES</b><br>
<pre>
http://mde-lab.aegean.gr/index.php/downloads
</pre>

<b>
2. Pap-smear Benchmark Data For Pattern Classiﬁcation<br></b>
Jan Jantzen, Jonas Norup , George Dounias , Beth Bjerregaard<br>

<pre>
https://www.researchgate.net/publication/265873515_Pap-smear_Benchmark_Data_For_Pattern_Classification

</pre>
<b>
3. Deep Convolution Neural Network for Malignancy Detection and Classification in Microscopic Uterine Cervix Cell Images</b><br>
Shanthi P B,1 Faraz Faruqi, Hareesha K S, and Ranjini Kudva<br>
<pre>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7062987/
</pre>

<b>
4. DeepCyto: a hybrid framework for cervical cancer classification by using deep feature fusion of cytology images</b><br>
Swati Shinde, Madhura Kalbhor, Pankaj Wajire<br>
<pre>
https://www.aimspress.com/article/doi/10.3934/mbe.2022301?viewType=HTML#b40
</pre>


