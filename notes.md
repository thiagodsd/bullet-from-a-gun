# Development Notes

## Notes on Object Detection Models

- Region Proposal Methods
    - Selective Search
    - Edge Boxes
    - Region Proposal Networks (RPN)
    - Superpixels

<br/>

- Dataset Preparation
- Neural Network Architecture Selection
  - Propósal (Two-Stage)
    - RCNN
    - Fast RCNN
    - Faster RCNN
    - Mask RCNN
    - RFCN
  - Proposal-Free (One-Stage)
    - YOLO
    - SSD
- Model Training
- Inference
- Evaluation
- Results

<br/>

YOLO vs RCNN
- YOLO is faster than RCNN
- RCNN is more accurate than YOLO
- YOLO is better for real-time applications

## Notes on Object Detection Metrics

+ IoU, intersection over union, area of overlap divided by area of union
+ AP, average precision: varying different thresholds for the IoU
+ AP50, average precision at 50% IoU
+ AP75, average precision at 75% IoU, it is more strict
+ APs, APm, APl, average precision for small, medium, large objects

In recent years, the most frequently used evaluation for detection is "Average Precision (AP)", which was originally introduced in VOC2007. AP is defined as the average detection precision under different recalls and is usually evaluated in a category-specific manner. The mean AP (mAP) averaged over all categories is typically used as the final metric of performance. To measure object localization accuracy, the Intersection over Union (IoU) between the predicted box and the ground truth is used to verify whether it is greater than a predefined threshold, such as 0.5. If it is, the object is identified as "detected"; otherwise, it is considered "missed". The 0.5-IoU mAP has then become the de facto metric for object detection​(nodes)​.

Citing/reference: `arXiv:1905.05055v3 [cs.CV] 18 Jan 2023`

## Development References
- [https://arxiv.org/abs/1807.05511](https://arxiv.org/abs/1807.05511)
- [https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298](https://www.sciencedirect.com/science/article/abs/pii/S1051200422004298)
- [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
-  https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETA/Fine_tuning_DETA_on_a_custom_dataset_(balloon).ipynb
-  https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb?ref=blog.roboflow.com#scrollTo=jbzTzHJW22up
-  [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/pdf/1801.05134)
-  [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)


```
    1  exit
    2  pwd
    3  exit
    4  git clone https://github.com/thiagodsd/bullet-from-a-gun.git
    5  pip install poetry
    6  ls
    7  cd bullet-from-a-gun/
    8  ls -lah
    9  cd ..
   10  python3 -m venv env_bullet
   11  source env_bullet/bin/activate
   12  cd bullet-from-a-gun/
   13  pip install poetry
   14  poetry install --no-root
   15  git pull
   16  kedro registry list
   17  git pull
   18  poetry install --no-root
   19  python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   20  pip install --upgrade pip
   21  python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   22  pip install 'git+https://github.com/facebookresearch/detectron2.git'
   23  nvcc --version
   24  nvidia-smi
   25  pip show torch
   26  cd ..
   27  git clone https://github.com/facebookresearch/detectron2.git
   28  python3 -m install -e detectron2
   29  python3 -m pip install -e detectron2
   30  python3 -m pip install -e detectron2 --user
   31  python -c "import torch; print(torch.version.cuda)"
   32  pip install torch torchvision torchaudio
   33  cd bullet-from-a-gun/
   34  python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
   35  git checkout yolo
   36  git pull
   37  git checkout yolo
   38  kedro registry list
   39  git pull
   40  poetry install --no-root
   41  kedro registry list
   42  kedro run -n yolo.yolov8_conf1_v1.hyperparameter_tuning_yolo
   43  pwd
   44  git pull
   45  kedro run -n yolo.yolov8_conf1_v1.hyperparameter_tuning_yolo
   46  git pull
   47  kedro run -n yolo.yolov8_conf1_v1.fine_tune_yolo,yolo.yolov8_conf1_v1.evaluate_yolo
   48  kedro run -n yolo.yolov8_conf1_v1.hyperparameter_tuning_yolo
   49  git pull
   50  kedro run -n yolo.yolov8_conf1_v1.hyperparameter_tuning_yolo
   51  kedro run -n yolo.yolov8_conf1_v1.fine_tune_yolo,yolo.yolov8_conf1_v1.evaluate_yolo
   52  git add -A
   53  git commit -m "experiment 1 @ ec2"
   54  git push origin yolo
   55  git pull
   56  kedro run -n yolo.yolov8_conf1_v1.evaluate_yolo
   57  ls -lah data/08_reporting/
   58  ls -lah data/08_reporting/plots/
   59  kedro run -n yolo.yolov8_conf2_v1.fine_tune_yolo,yolo.yolov8_conf2_v1.evaluate_yolo
   60  git add -A
   61  git commit -m "experiment 2"
   62  git push origin yolo
   63  kedro run -n yolo.yolov8_conf3_v1.fine_tune_yolo,yolo.yolov8_conf3_v1.evaluate_yolo
   64  git pull
   65  kedro run -n yolo.yolov8_conf3_v1.fine_tune_yolo,yolo.yolov8_conf3_v1.evaluate_yolo
   66  git add -A; git commit -m "experiment 3"; git push origin yolo
   67  git pull
   68  kedro run -n yolo.yolov8_conf4_v1.fine_tune_yolo,yolo.yolov8_conf4_v1.evaluate_yolo
   69  git pull
   70  kedro run -n yolo.yolov8_conf4_v1.fine_tune_yolo,yolo.yolov8_conf4_v1.evaluate_yolo
   71  kedro run -n yolo.yolov8_conf5_v1.fine_tune_yolo,yolo.yolov8_conf5_v1.evaluate_yolo
   72  kedro run -n yolo.yolov8_conf6_v1.fine_tune_yolo,yolo.yolov8_conf6_v1.evaluate_yolo
   73  git pull
   74  kedro run -n yolo.yolov8_conf6_v1.fine_tune_yolo,yolo.yolov8_conf6_v1.evaluate_yolo
   75  kedro run -n yolo.yolov8_conf7_v1.fine_tune_yolo,yolo.yolov8_conf7_v1.evaluate_yolo
   76  git add -A; git commit -m "experiments 4-7"; git push origin yolo
   77  git pull
   78  kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo
   79  git pull
   80  kedro run -n yolo.yolov8_conf8_v1.fine_tune_yolo
   81   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
   82  git pull
   83   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
   84  git pull
   85   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
   86  vi src/bullet_from_a_gun/pipelines/yolo/nodes.py 
   87   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
   88  git add -A; git commit -m "experiment 8"; git push origin yolo
   89  exit
   90  ls
   91  history
   92  git add -A; git commit -m "experiment 8"; git push origin yolo
   93  source env_bullet/bin/activate
   94  history
   95  cd bullet-from-a-gun/
   96  git pull
   97  git branch
   98  git fetch
   99  git checkout rcnn
  100  git pull
  101  git lok
  102  git log
  103  85   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  104     86  vi src/bullet_from_a_gun/pipelines/yolo/nodes.py 
  105     87   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  106     88  git add -A; git commit -m "experiment 8"; git push origin yolo
  107  85   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  108     86  vi src/bullet_from_a_gun/pipelines/yolo/nodes.py 
  109     87   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  110     88  git add -A; git commit -m "experiment 8"; git push origin yolo
  111  85   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  112     86  vi src/bullet_from_a_gun/pipelines/yolo/nodes.py 
  113     87   kedro run -n yolo.yolov8_conf8_v1.hyperparameter_tuning_yolo
  114     88  git add -A; git commit -m "experiment 8"; git push origin yolo
  115  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  116  cd ..
  117  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  118  cd detectron2/
  119  ls
  120  history | grep detectron
  121  python3 -m install -e detectron2
  122  python3 -m pip install -e detectron2
  123  cd ..
  124  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --user
  125  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  126  pip show torch
  127  sudo yum install gcc gcc-c++
  128  python -m pip install --use-pep517 'git+https://github.com/facebookresearch/detectron2.git'
  129  pip install flash-attn --no-build-isolation
  130  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  131  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
  132  cd detectron2/
  133  pip install
  134  pip help install
  135  python setup.
  136  python setup.py 
  137  python setup.py install
  138  pip install torchvision
  139  cd ..
  140  source deativate
  141  source deactivate
  142   deactivate
  143  poetry init
  144  poetry add kedro==0.19.6 torch torchvision
  145  conda create --name detectron2 python==3.9 -y
  146  ls -alh
  147  rm -rf det
  148  rm -rf detectron2
  149  ls
  150  ls -lah
  151  rm -rf env_bullet/
  152  su -kh
  153  du -kh
  154  conda create --name detectron2 python==3.9 -y
  155  conda activate detectron2
  156  pip install torch torchvision
  157  rm -rf bullet-from-a-gun/
  158  pip install torch torchvision
  159  ls -lah
  160  rm -rf .cache/
  161  deactivate
  162  conda deactivate
  163  pip uninstall torch
  164  pip uninstall torchvision
  165  exit
  166  df -h
  167  ls -lah
  168  exit
  169  ls -lah
  170  history
  171  git clone https://github.com/thiagodsd/bullet-from-a-gun.git
  172  conda activate detectron2
  173  conda init
  174  conda activate detectron2
  175  pip install torch torchvision
  176  df - kh
  177  df -kh
  178  conda install -c conda-forge pybind11
  179  conda install -c conda-forge gxx
  180  conda install -c anaconda gcc_linux-64
  181  conda upgrade -c conda-forge --all
  182  pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
  183  sudo yum install gcc11-c++
  184  sudo amazon-linux-extras install epel
  185  sudo amazon-linux-extras install gcc9
  186  sudo yum install -y epel-release
  187  sudo yum groupinstall -y "Development Tools"
  188  sudo yum install -y gcc gcc-c++ kernel-devel
  189  wget https://ftp.gnu.org/gnu/gcc/gcc-9.3.0/gcc-9.3.0.tar.gz
  190  tar -xzf gcc-9.3.0.tar.gz
  191  cd gcc-9.3.0
  192  ./contrib/download_prerequisites
  193  mkdir build && cd build
  194  ../configure --enable-languages=c,c++ --disable-multilib
  195  make -j$(nproc)
  196  sudo make install
  197  gcc --version
  198  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  199  gcc --version
  200  cd ..
  201  sudo apt update
  202  sudo amazon-linux-extras install epel -y
  203  sudo yum install -y centos-release-scl
  204  sudo yum install -y devtoolset-9
  205  scl enable devtoolset-9 bash
  206  gcc --version
  207  export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc
  208  export CXX=/opt/rh/devtoolset-9/root/usr/bin/g++
  209  export PATH=/opt/rh/devtoolset-9/root/usr/bin:$PATH
  210  echo $CC
  211  echo $CXX
  212  conda install -c conda-forge gxx_linux-64  # Provides GNU C++ compiler in Conda
  213  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  214  pip install 'git+https://github.com/facebookresearch/detectron2.git'history
  215  [Ahist
  216  history
  217  nvcc --version
  218  gcc --version
  219  conda deactivate
  220  conda -gh
  221  conda -h
  222  conda remove detectron2
  223  conda list
  224  ls
  225  cd ..
  226  conda list
  227  ld
  228  ls
  229  ls -lah
  230  ls -ah
  231  conda uninstall --name detectron2
  232  conda remove --name detectron2
  233  conda remove -v
  234  conda remove -h
  235  conda remove -n detectron2
  236  conda env remove -n detectron2
  237  gcc --version
  238  conda create --name detectron2 python==3.9 -y
  239  gcc --version
  240  conda activate detectron2
  241  gcc --version
  242  pip install torch torchvision
  243  pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
  244  git checkout detectron2
  245  cd bullet-from-a-gun/
  246  git checkout detectron2
  247  git pull
  248  pip install kedro==0.19.6
  249  pip install kedro-viz==8.0.1 kedro-datasets[pandas]==2.1.0
  250  pip install kedro-datasets[matplotlib]==2.1.0
  251  git add -A; git commit -m "experiment 8"; git push origin yolo
  252  history
  253  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  254  pip install cv2
  255  pip install opencv-python==4.9.0.80
  256  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  257  pip install poetry
  258  poetry lock
  259  poetry install --no-install
  260  poetry install --no-root
  261  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  262  poetry add ERROR    An error has occurred: module 'PIL.Image' has no attribute 'LINEAR'     
  263  poetry add pillow==9.5.0
  264  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  265  git checkout rcnn
  266  git pull
  267  git add -A
  268  git commit -m "."
  269  git checkout rcnn
  270  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  271  poetry lock
  272  poetry install --no-root
  273  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  274  poetry add pillow==9.5.0
  275  kedro run -n detectron2.rccn_101_conf1_v1.fine_tune_detectron2
  276  kedro run -n detectron2.rccn_101_conf1_v1.evaluate_detectron2
  277  kedro run -n detectron2.rccn_101_conf2_v1.fine_tune_detectron2
  278  kedro run -n detectron2.rccn_101_conf2_v1.evaluate_detectron2
  279  kedro run -n detectron2.rccn_101_conf3_v1.fine_tune_detectron2
  280  kedro run -n detectron2.rccn_101_conf3_v1.evaluate_detectron2
  281  history
  282  kedro run -n detectron2.rccn_101_conf4_v1.fine_tune_detectron2
  283  kedro run -n detectron2.rccn_101_conf4_v1.evaluate_detectron2
  284  kedro run -to-nodes=detectron2.rccn_101_conf4_v1.evaluate_detectron2
  285  kedro run --to-nodes=detectron2.rccn_101_conf4_v1.evaluate_detectron2
  286  kedro run -n detectron2.rccn_101_conf4_v1.fine_tune_detectron2
  287  kedro run --to-nodes=detectron2.rccn_101_conf4_v1.evaluate_detectron2
  288  kedro run -n detectron2.rccn_101_conf4_v1.evaluate_detectron2
  289  kedro run -n detectron2.rccn_101_conf5_v1.fine_tune_detectron2
  290  sudo yum install -y htop
  291  kedro run -n detectron2.rccn_101_conf5_v1.fine_tune_detectron2
  292  kedro run -n detectron2.rccn_101_conf5_v1.evaluate_detectron2
  293  kedro run -n detectron2.rccn_101_conf4_v1.fine_tune_detectron2
  294  kedro run -n detectron2.rccn_101_conf4_v1.evaluate_detectron2
  295  git add -A
  296  git commit -m "rcnn experiments finished"
  297  git push origin rcnn
  298  kedro run -n detectron2.rccn_101_conf6_v1.fine_tune_detectron2
  299  kedro run -n detectron2.rccn_101_conf6_v1.evaluate_detectron2
  300  history
```