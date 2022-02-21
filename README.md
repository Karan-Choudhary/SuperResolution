## Super Resolution using GANs
* SRGAN is an image-to-image translation model using [Generative Adversarial Networks](https://arxiv.org/abs/1609.04802)
* This project is developed with [DVC](https://dvc.org/) pipeline
* Dataset used - [DIV2K - bicubic downscaling x8 competition](https://data.vision.ee.ethz.ch/cvl/ntire17//)
* Model is only trained for 35 epochs on Google Colab
* You can train it according to your need by changing parameters in [params.yaml](https://github.com/Karan-Choudhary/SuperResolution/blob/main/params.yaml) file
* Model Architecture:
![model](https://user-images.githubusercontent.com/54716931/154962938-59e6f2e3-93e7-4c4c-960a-c740e4b0d8e6.jpeg)

### Requirements
* [Create a new virtual environment](https://docs.python.org/3/library/venv.html).
* Python 3.6 or greater.
* DVC
* Run Command-
```
pip install -r requirements.txt
```

### Usage:
* Change parameters and directories according to your system (recommendation - do not change) in [params.yaml](https://github.com/Karan-Choudhary/Sketchs_to_ColorImages/blob/main/params.yaml).
* Setup the Directory named ***Results*** in the root directory.
* Make sure you have DVC initialized in the root directory</br>
You can do it with the command:
```
dvc init
```
* Add **data\train** & **data\val** for data tracking in dvc by using command:
For Windows-
```
dvc add data\train\HR
dvc add data\train\LR
dvc add data\validation\HR
dvc add data\validation\LR
```
For Linux-
```
dvc add data/train/HR
dvc add data/train/LR
dvc add data/validation/HR
dvc add data/validation/LR
```
* Finally, run:
```
dvc repro
```
* Trained models will be saved in ***saved_models*** directory.
* Results will be saved in ***Results\present_datatime*** directory with name **present_time_result.png**.
 
Warning: Do not try to change **dvc.lock file** and ***.dvc directory***
