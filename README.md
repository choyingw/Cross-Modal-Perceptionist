# Cross-Modal Perceptionist
CVPR 2022 "Cross-Modal Perceptionist: Can Face Geometry be Gleaned from Voices?"

Cho-Ying Wu, Chin-Cheng Hsu, Ulrich Neumann, University of Southern California

[<a href="">Paper (soon!)</a>] [<a href="https://choyingw.github.io/works/Voice2Mesh/index.html">Project page</a>] [<a href="">Data (soon!)</a>]

<img src="demo/overall_purpose.png">

We study the cross-modal learning and analyze the correlation between voices and 3D face geometry. Unlike previous methods for studying this correlation between voices and faces and only work on  the 2D domain, we choose 3D representation that can better validate the supportive evidence from the physiology of the correlation between voices and skeletal and articulator structures, which potentially affect facial geometry.

Comparison of recovered 3D face meshes with the baseline.

<img src="demo/supervised_gt.png">


Consistency for the same identity using different utterances.

<img src="demo/coherence.png"> 

**Demo codes**

We test on Ubuntu 16.04 LTS, NVIDIA 2080 Ti (only GPU is supported), and use anaconda for installing packages

Install packages

1. `conda create --name CMP python=3.8`
2. Install pytorch compatible to your computer, we test on pytorch v1.7.1 (should be compatible with other 1.0+ versions)
3. install other dependency: opencv-python, scipy, PIL, Cython

    Or use the environment.yml we provide instead: 
    - `conda env create -f environment.yml`
    - `conda activate CMP`

4. Build the rendering toolkit (by c++ and cython) for overlapping 3D meshes on images with configurations

    ```
    cd Sim3DR
    bash build_sim3dr.sh
    cd ..
    ```

Download pretrained models and 3DMM configuration data

5. Download from [<a href="https://drive.google.com/file/d/1tqTSDrVVL3LkOWN-hduELm3YkWJ2ZUqu/view?usp=sharing">here</a>] (~160M) and unzip under the root folder

Run

6. `python demo.py` (This will fetch the preprocessed MFCC and use them as network inputs)
7. Results will be generated under `data/results/` (pre-generated references are under `data/results_reference`)


This project is developed on [<a href="https://github.com/choyingw/SynergyNet">SynergyNet</a>], [<a href="https://github.com/cleardusk/3DDFA_V2">3DDFA-V2</a>] and [<a href="https://github.com/cmu-mlsp/reconstructing_faces_from_voices">reconstruction-faces-from-voice</a>]
