<p align=center>
    <img src="img/7-balls-new.gif">
</p>

<p align=center>
    <a target="_blank" href="https://travis-ci.com/chonyy/ML-auto-baseball-pitching-overlay" title="Build Status"><img src="https://travis-ci.com/chonyy/ML-auto-baseball-pitching-overlay.svg?branch=master"></a>
    <a target="_blank" href="#" title="language count"><img src="https://img.shields.io/github/languages/count/chonyy/ML-auto-baseball-pitching-overlay"></a>
    <a target="_blank" href="#" title="top language"><img src="https://img.shields.io/github/languages/top/chonyy/ML-auto-baseball-pitching-overlay?color=orange"></a>
    <a target="_blank" href="https://img.shields.io/github/pipenv/locked/python-version/chonyy/daily-nba" title="Python version"><img src="https://img.shields.io/github/pipenv/locked/python-version/chonyy/daily-nba?color=green"></a>
    <a target="_blank" href="https://opensource.org/licenses/MIT" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a target="_blank" href="#" title="repo size"><img src="https://img.shields.io/github/repo-size/chonyy/ML-auto-baseball-pitching-overlay"></a>
    <a target="_blank" href="http://makeapullrequest.com" title="PRs Welcome"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
</p>

> ⚾ Automatically overlaying pitch motion and trajectory with machine learning!

This work is now published in [ACM ICMR 2021](https://dl.acm.org/doi/abs/10.1145/3460426.3463576)!

This project takes your baseball pitching clips and **automatically** generates the overlay. The input pitching clip could be directly from your phone or camera. The release point will be automatically detected by the program. This system will trace the trajectory and align all the videos to generate the overlay.

A fine-tuned YOLOv4 model is used to get the location of the ball. Then, I implemented SORT tracking algorithm to keep track of each individual ball. Lastly, I have applied some image registration techniques to deal with slight camera shift on each clip.

I'm still trying to improve it! Feel free to follow this project, also check out the Todo list.

## 💻 Getting Started

These instructions will get you a copy of the project, and generates your own pitching overlay clip!

### Get a copy

Get a copy of this project by simply running the git clone command.

```git
git clone https://github.com/chonyy/ML-auto-baseball-pitching-overlay.git
```

### Prerequisites

Before running the project, we have to install all the dependencies from requirements.txt

```pip
pip install -r requirements.txt
```

### Overlay!

Last, run the project with your own clips!

#### Try a sample

```python
python pitching_overlay.py
```

#### Try with yout own clips

Place your pitching videos in a folder, then specify the path in the CLI.

```python
python pitching_overlay.py --videos_folder "./videos/videos"
```

## 🔨 Project Structure

<p align=center>
    <img src="img/structure-new.png">
</p>

## 🎬 More Demo

<p align=center>
    <img src="img/2_balls_new.gif">
</p>
<p align=center>
    <img src="img/4-balls.gif">
</p>
<p align=center>
    <img src="img/3-balls-new.gif">
</p>
<p align=center>
    <img src="img/3-balls-diff.gif">
</p>

## ☑️ Todo

-   [x] Implement image registration to deal with camera shift
-   [ ] Build a demo web app for people to use it in realtime on web
-   [ ] Enable custom parameter tuning
-   [ ] Improve the visual effect
-   [ ] Write a Medium post to explain the technical workflow
-   [ ] Draw a structure diagram
