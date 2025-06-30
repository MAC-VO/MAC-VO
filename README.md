# <div align="center">MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry</div>

### <div align="center">🥇 ICRA 2025 Best Conference Paper Award<br/>🥇 ICRA 2025 Best Paper Award on Robot Perception</div>

<p align="center">
  <a href="https://mac-vo.github.io"><img src="https://img.shields.io/badge/Homepage-4385f4?style=flat&logo=googlehome&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2409.09479v2"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=O_HowJk-GDw"><img src="https://img.shields.io/badge/YouTube-b31b1b?style=flat&logo=youtube&logoColor=white"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


<p align="center">
  <img src="asset/ICRAvideo.gif" alt="ICRA floor 3" width="600" />
</p>


> [!NOTE]  
> We plan to release TensorRT accelerated implementation and adapting more matching networks for MAC-VO. If you are interested, please star ⭐ this repo to stay tuned.

> [!NOTE]
>
> We provide **[documentation for extending MAC-VO](https://mac-vo.github.io/wiki/)** for extending MAC-VO or using this repository as a boilerplate for *your* learning-based Visual Odometry.
>

## 🔥 Updates

* [Jun 2025] We release the **MAC-VO Fast Mode** - with faster pose graph optimization and mixed-precision inference, we achieve 2x speedup compare to previous version and reach speed of 12.5fps on 480x640 images. 

  See `Config/Experiment/MACVO/MACVO_Fast.yaml` for detail. 
  
  Original example is also boosted from 5fps to 7fps and the config file is moved to `MACVO_Performant.yaml`.
* [Apr 2025] Our work was nominated as the **ICRA 2025 Best Paper Award Finalist** (top 1%)! Keep an eye on our presentation on May 20, 16:35-16:40 Room 302. We also plan to provide a real-world demo at the conference.
* [Mar 2025] We boost the performance of MAC-VO with a new backend optimizer, the MAC-VO now also supports *dense mapping* without any additional computation.
* [Jan 2025] Our work is accepted by the IEEE International Conference on Robotics and Automation (ICRA) 2025. We will present our work at ICRA 2025 in Atlanta, Georgia, USA.
* [Nov 2024] We released the ROS-2 integration at https://github.com/MAC-VO/MAC-VO-ROS2 along with the documentation at https://mac-vo.github.io/wiki/ROS/

## Download the Repo

Clone the repository using the following command to include all submodules automatically.

`git clone https://github.com/MAC-VO/MAC-VO.git --recursive`


## 📦 Installation & Environment

### Environment

1. **Docker Image**

    ```bash
    $ docker build --network=host -t macvo:latest -f Docker/Dockerfile .
    ```

2. **Virtual Environment**

    You can setup the dependencies in your native system. MAC-VO codebase can only run on Python 3.10+. See `requirements.txt` for environment requirements.

    <details>
      <summary>How to adapt MAC-VO codebase to Python &lt; 3.10?</summary>
      
      The Python version requirement we required is mostly due to the [`match`](https://peps.python.org/pep-0634/) syntax used and the [type annotations](https://peps.python.org/pep-0604/).

      The `match` syntax can be easily replaced with `if ... elif ... else` while the type annotations can be simply removed as it does not interfere runtime behavior.
    </details>

### Pretrained Models

All pretrained models for MAC-VO, stereo TartanVO and DPVO are in our [release page](https://github.com/MAC-VO/MAC-VO/releases/tag/model). Please create a new folder `Model` in the root directory and put the pretrained models in the folder.

    $ mkdir Model
    $ wget -O Model/MACVO_FrontendCov.pth https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth
    $ wget -O Model/MACVO_posenet.pkl https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl

## 🚀 Quick Start: Run MAC-VO on Demo Sequence

Test MAC-VO immediately using the provided demo sequence. The demo sequence is a selected from the TartanAir v2 dataset.

### 1/4 Download the Data

1. Download a demo sequence through [Google Drive](https://drive.google.com/file/d/1kCTNMW2EnV42eH8g2STJHcVWEbVKbh_r/view?usp=sharing).
2. Download pre-trained model for [frontend model](https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth) and [posenet](https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl).

### 2/4 Start the Docker
To run the Docker: 

    $ docker run --gpus all -it --rm  -v [DATA_PATH]:/data -v [CODE_PATH]:/home/macvo/workspace macvo:latest

To run the Docker with visualization: 

    $ xhost +local:docker; docker run --gpus all -it --rm  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v [DATA_PATH]:/data -v [CODE_PATH]:/home/macvo/workspace macvo:latest


### 3/4 Run MAC-VO

We will use `Config/Experiment/MACVO/MACVO_example.yaml` as the configuration file for MAC-VO.

1. Change the `root` in the data config file 'Config/Sequence/TartanAir_example.yaml' to reflect the actual path to the demo sequence downloaded.
2. Run with one of the following command:

    *Performant Mode* - best performance with moderate speed (7.5fps on 480x640 image)

    ```bash
    $ cd workspace
    $ python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/TartanAir_example.yaml
    ```

    *Fast Mode* - slightly degraded performance (~5% increase in RTE and ROE) with most speed (12.5fps on 480x640 image)

    ```bash
    $ cd workspace
    $ python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/TartanAir_example.yaml
    ```

> [!NOTE]
>
> See `python MACVO.py --help` for more flags and configurations.

### 4/4 Visualize and Evaluate Result

Every run will produce a `Sandbox` (or `Space`). A `Sandbox` is a storage unit that contains all the results and meta-information of an experiment. The evaluation and plotting script usually requires one or more paths of sandbox(es).

#### **Evaluate Trajectory**

  Calculate the absolute translate error (ATE, m); relative translation error (RTE, m/frame); relative orientation error (ROE, deg/frame); relative pose error (per frame on se(3)).

  ```bash
  $ python -m Evaluation.EvalSeq --spaces SPACE_0, [SPACE, ...]
  ```

#### **Plot Trajectory**

  Plot sequences, translation, translation error, rotation and rotation error.

  ```bash
  $ python -m Evaluation.PlotSeq --spaces SPACE_0, [SPACE, ...]
  ```

## 🛠️ Additional Commands and Utility

* **Run MAC-VO (*Ours* method) on a Single Sequence**
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
    ```

* **Run MAC-VO for Ablation Studies**
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/Ablation_Study/[CHOOSE_ONE_CFG].yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
    ```

* **Run MAC-VO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_MACVO --odom [PATH_TO_ODOM_CONFIG]
  ```

* **Run MAC-VO Mapping Mode**

  ```bash
  $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO_MappingMode.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
  ```

### 📊 Plotting and Visualization

We used [the Rerun](https://rerun.io) visualizer to visualize 3D space including camera pose, point cloud and trajectory.

* **On Machine with GUI**

  1. Run `MACVO.py` with the following command line
    
        ```bash
        $ python MACVO.py --useRR --odom [ODOM_CONFIG] --data [DATA_CONFIG]
        ```
     
        A rerun visualizer should pop up with the trajectory and *per-frame* point cloud & tracking features visualized.
  2. To accumulate the point cloud for dense mapping visualization, please follow the instruction here: https://github.com/MAC-VO/MAC-VO/issues/4#issuecomment-2495620352

* **On Headless Machine**

  1. Install the `rerun_sdk` python package on both your machine (with GUI) and remote headless environment. Also setup a port forwarding from remote port `9877` to your local machine port `9877`.
  2. Start a rerun server by rerun --serve & on the headless machine
  3. On your machine (with GUI), run rerun ws://localhost:9877 to connect to the remote visualization server. You should see "2 sources connected" on the top right corner of visualizer if everything works smoothly.
  4. On the headless machine, run
      ```bash
      $ python MACVO.py --useRR --odom [ODOM_CONFIG] --data [DATA_CONFIG]
      ```
  5. To accumulate the point cloud for dense mapping visualization, please follow the instruction here: https://github.com/MAC-VO/MAC-VO/issues/4#issuecomment-2495620352

### 📈 Baseline Methods

We also integrated two baseline methods (DPVO, TartanVO Stereo) into the codebase for evaluation, visualization and comparison.

<details>
<summary>
Expand All (2 commands)
</summary>

* **Run DPVO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_DPVO --odom ./Config/Experiment/Baseline/DPVO/DPVO.yaml
  ```

* **Run TartanVO (Stereo) on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_TartanVO --odom ./Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml
  ```

</details>


## 📐 Coordinate System in this Project

**PyTorch Tensor Data** - All images are stored in `BxCxHxW` format following the convention. Batch dimension is always the first dimension of tensor.

**Pixels on Camera Plane** - All pixel coordinates are stored in `uv` format following the OpenCV convention, where the direction of uv are "east-down". *Note that this requires us to access PyTorch tensor in `data[..., v, u]`* indexing.

**World Coordinate** - `NED` convention, `+x -> North`, `+y -> East`, `+z -> Down` with the first frame being world origin having identity SE3 pose.


## 🤗 Customization, Extension and Future Developement

> This codebase is designed with *modularization* in mind so it's easy to modify, replace, and re-configure modules of MAC-VO. One can easily use or replase the provided modules like flow estimator, depth estimator, keypoint selector, etc. to create a new visual odometry.

We welcome everyone to extend and redevelop the MAC-VO. For documentation please visit the [Documentation Site](https://mac-vo.github.io/wiki/)
