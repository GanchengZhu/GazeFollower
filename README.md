<div align="center">
  <a href="https://github.com/GanchengZhu/GazeFollower">
    <img width="160" height="160" src="https://raw.githubusercontent.com/GanchengZhu/GazeFollower/main/gazefollower/res/image/gazefollower.png">
  </a>

  <h1>GazeFollower</h1>

<b>An open-source gaze tracking system for web cameras</b><br/>
<i>Simple, Fast, Pythonic, Accurate</i><br/>

<p>
    <img src="https://img.shields.io/github/languages/top/ganchengzhu/gazefollower" alt="Top language">
    <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
    <img src="https://img.shields.io/codacy/grade/e21ccd9e469d4b9abd69efeaaa587cc2" alt="Code quality">
</p>
</div>

## Introduction

**GazeFollower** is a powerful and easy-to-use gaze tracking system designed specifically for use with web cameras. It
offers an intuitive Python API, allowing developers and researchers to integrate gaze tracking into their projects with
minimal setup. GazeFollower provides tools for real-time gaze tracking, calibration, and data recording, making it ideal
for applications in psychology, usability testing, and more.

## Features

- **Accurate Tracking**: Achieves high accuracy and precision with built-in calibration methods.
- **Pythonic API**: Easy-to-use, with functions for common tasks like calibration and data saving.
- **Lightweight & Fast**: Optimized for real-time performance, ensuring smooth operation on most systems.
- **Experiment Ready**: Includes methods for triggering and saving data, ideal for experiment-based applications.

## Installation

You can install GazeFollower via pip or by cloning the repository.

### Installing with pip

```bash
python -m pip install gazefollower
```

### Git clone from Github

```bash
git clone https://github.com/GanchengZhu/GazeFollower
cd GazeFollower
python setup.py install
```

## Quick Start

Here's a basic example of how to use GazeFollower:

```python
# _*_ coding: utf-8 _*_

import pygame
from gazefollower import GazeFollower

gaze_follower = GazeFollower()

gaze_follower.preview()
gaze_follower.calibrate()

gaze_follower.start_sampling()
# your experiment code here
gaze_follower.send_trigger(10)
pygame.time.wait(5)
# your experiment code here
gaze_follower.stop_sampling()
gaze_follower.save_data("demo.csv")
gaze_follower.release()
```

More detailed usage information can be found [here](MORE_INFO.md).

## New feature

Using the BlazeFace model for face alignment to reduce inference time.

```python
gf = GazeFollower(camera=WebCamCamera(webcam_id=0),
                  face_alignment=BlazeFaceAlignment())
```

## Note

This depository contains a model train on 7 million images that is sufficient for most applications. 

To gain access to the base model trained on 32 million images, please email zhiguo@zju.edu.cn.

### Email Prompt

Here’s a template for your request email. Please keep the subject line unchanged:

```
Subject: Request for Access to the Base Model Trained on 32 Million Images

Dear Prof. Zhiguo Wang,

I hope this message finds you well.

My name is [Your Name], and I am a [student/researcher] at [Your Affiliation]. I am writing to request access to the base model trained on 32 million images.

I assure you that I will use this model solely for academic and research purposes and will not utilize it for commercial activities or share it with others.

Thank you for considering my request. I look forward to receiving access to the model.

Best regards,
[Your Name]
```

### The Usage of The Base Model Trained on 32 Million Images

```python
import pygame
from gazefollower import GazeFollower
from gazefollower.gaze_estimator import MGazeNetGazeEstimator

# The base model path need to pass here
gaze_follower = GazeFollower(gaze_estimator=MGazeNetGazeEstimator(model_path='path to model'))

gaze_follower.preview()
gaze_follower.calibrate()

gaze_follower.start_sampling()
# your experiment code
gaze_follower.send_trigger(10)
pygame.time.wait(5)
# your experiment code
gaze_follower.stop_sampling()

gaze_follower.save_data("demo.csv")
gaze_follower.release()
```

## How to fine-tune

Please send an email to `zhiguo@zju.edu.cn` to access data collection, data preprocessing, and fine-tuning code

## License Information

This project is licensed under
the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Please see LICENSE file.

### Disclaimer

The work is provided "as-is" without any warranties, express or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose, or non-infringement.


### Citations

```BibTeX
@article{zhu2025gazefollower,
  title={GazeFollower: An open-source system for deep learning-based gaze tracking with web cameras},
  author={Zhu, Gancheng and Duan, Xiaoting and Huang, Zehao and Wang, Rong and Zhang, Shuai and Wang, Zhiguo},
  journal={Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  volume={8},
  number={2},
  pages={1--18},
  year={2025},
  publisher={ACM New York, NY}
}
```

```BibTeX
@article{zhu2024neural,
  title={Neural Networks With Linear Adaptive Batch Normalization and Swarm Intelligence Calibration for Real-Time Gaze Estimation on Smartphones},
  author={Zhu, Gancheng and Li, Yongkai and Zhang, Shuai and Duan, Xiaoting and Huang, Zehao and Yao, Zhaomin and Wang, Rong and Wang, Zhiguo},
  journal={International Journal of Intelligent Systems},
  volume={2024},
  number={1},
  pages={2644725},
  year={2024},
  publisher={Wiley Online Library}
}
```

```BibTeX
@article{zhu2025smartphone,
  title={Smartphone eye-tracking with deep learning: Data quality and field testing},
  author={Zhu, Gancheng and Huang, Zehao and Duan, Xiaoting and Zhang, Shuai and Wang, Rong and Li, Yongkai and Wang, Zhiguo},
  journal={Behavior Research Methods},
  volume={57},
  number={7},
  pages={1--19},
  year={2025},
  publisher={Springer}
}
```

# NEED TO IMPLEMENTATION

- Blink Detection
- Event Detection
- Calibration Optimization
- Reduce Jitter
