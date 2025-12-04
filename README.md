<sup>This code is written by [Oliver Hausdörfer](https://oliver.hausdoerfer.de/), [Martin Schuck](https://github.com/amacati), [Luca Worbis](https://github.com/cryxil), [Yi Lu](https://github.com/yilutum), [Yufei Hua](https://github.com/yufei4hua), and [Timo Class](https://github.com/clsti). For details please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md).<sup>

# Control for Robotics - From Optimal Control to Decision Making <small><small>Advanced Programming Exercises</small></small>


### **[Book](https://utiasdsl.github.io/cfr/)**: Control for Robotics - From Optimal Control to Decision Making (Angela P. Schoellig, SiQi Zhou)

### **[Course](https://www.ce.cit.tum.de/lsy/teaching/advanced-robot-learning-and-decision-making/)**: ARLDM - Advanced Robot Learning and Decision Making (TUM 0CIT433037)

> 🚀 Welcome! To start the exercises read the following instructions carefully.

> ❗ **Students taking the course for Uni must use the code provided directly through course and ARTEMIS!**

![example](example_1.gif)
![example](example_2.gif)

## 🚀 Introduction
This code accompanies the drone case study from our book [Control for Robotics: From Optimal Control to Reinforcement Learning](https://utiasdsl.github.io/cfr/). We also use this code base as a foundation for our Advanced Course [Advanced Robot Learning and Decision Making](https://www.ce.cit.tum.de/lsy/teaching/advanced-robot-learning-and-decision-making/) at TUM (0CIT433037).

The code is designed as programming exercises, where you are guided to implement your own controllers:
- Exercise 01: Introduction to code and frameworks
- Exercise 02: LQR, ILQR
- Exercise 03: MPC
- Exercise 04: GP-MPC
- Exercise 05: Model Learning
- Exercise 06: DRL
The code includes test cases, to test your implementations locally.

The tasks evolve around controlling a simulated drone - from reaching a simple goal pose to follow complex trajectories.

# Using the Programming Exercises

## ⚙️ Preliminaries and Setup

- You require at least 20GB of free permament storage and 8GB (preferably 16GB) of RAM on your device.
- We strongly recommend using Linux. It is the most widely used OS in robotics. We support running the exercises in our  VS Code Dev Container on Linux (recommended) and Windows 11 (WSL2). You can also setup the environment with manual installation, but you may encounter installation issues. For MacOS, most students use a VM with Ubuntu (recommended) or manual installation.
- For any setup issues **check the `Common Issues` section below** before reaching out for help.
- If you find **bugs or issues with the code**, please open a issue or pull request here in the public [GitHub repository](https://github.com/utiasDSL/ARLDM-Advanced-Robot-Learning-And-Decision-Making/edit/main/README.md).

## 🛠️ Used Tools
We will use:
- <kbd>Git</kbd> as version control system: Find a [Git introduction here](https://docs.duckietown.com/ente/devmanual-software/basics/development/git.html). Specifically, you need to know how to `clone`, `add`, `commit`, and `push`.
- <kbd>Python</kbd> as programming language: Find a [Python tutorial here](https://www.tutorialspoint.com/python/index.htm), in particular make sure to go through the basic data structures (tuples, lists, dictionaries, sets,…), loops (while, for,…), conditional statements (if-else), functions, classes and objects.
- <kbd>Docker</kbd> as environment containerization (but you won’t see it much). A [container]((https://www.docker.com/resources/what-container/)) is a standard unit of software that packages up all required code and its dependencies. <kbd>[VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)</kbd> allows us to ship such a container as a full-featured development environment that you can readily use for our exercises. Containerization is ubiquitous in modern software development: find a [Docker introduction here](https://docs.duckietown.com/ente/devmanual-software/basics/development/docker.html).
- <kbd>[VS Code](https://code.visualstudio.com/)</kbd>: Visual Studio Code provides a set of tools that speed up software development (debugging, testing, ...). Moreover, we will provide environment configurations that setup the exercise' container and make your life easier.


If they all sound completely new to you do not panic. We will require a very basic use of most of them, but it is a good time to start learning these tools since they are all widely adopted in modern robotics.

## 👨‍💻 Setting up the exercises
The following are the usual steps involved in setting up VS Code Devcontainers. One special feature is that we render simulations directly on the container host's display. Such display forwarding is a common failure case, is the reason why the exercise container does not work on MacOS for the moment, and explains all of the more special instructions below.

### Linux (recommended)
1. Make sure you are using a X11 Desktop session (not wayland): https://askubuntu.com/a/1516672.
2. Install [Docker](https://docs.docker.com/engine/install/), and make sure you can run Docker hello-world without error: `docker run hello-world`.
3. Install [VS Code](https://code.visualstudio.com/), with [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [container tools extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers).
4. Now your local computer is setup to run the exercise container.
5. In *this project*, rename `/.devcontainer/devcontainer.linux.json` to `/.devcontainer/devcontainer.json`.
6. Open *this project* in VS code (Select File -> Open Folder). VS code should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open it manually. **Note**: Opening the container for the first time might take up to 15 min.
8. Now you should be ready to start the exercises.

### Windows 11 (WSL2)
For windows, you require [WSL2](https://learn.microsoft.com/de-de/windows/wsl/install) to run the devcontainer, which is actually a Linux within Windows.  Here are the important steps:

1. Follow the [official installation steps (under Getting started)](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_getting-started) to install VS Code Devcontainers in WSL2 and Docker.
   - **Note 1:** install Ubuntu 22.04 or above
   - **Note 2:** if you didnt get prompted to `enable WSL integration` by Docker as written in the above installation steps, open `Docker Desktop`, and navigate to the settings. Here, manually enable WSL integration. (There are TWO setting options for this. Make sure to enable BOTH!)
   - Make sure you can run Docker hello-world without error: `docker run hello-world`
<!--3. Install [VSCode](https://code.visualstudio.com/), with the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl), [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [remote dev pack](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).-->
3. Make sure you have *this exercise code* cloned in the WSL2 file system (and not in the windows file system), for instance to `/home` (`~`). (Performance when working on the WSL file system is much better compared to Windows file system). You can access the WSL filesystem by starting a WSL2 / Ubuntu terminal.
4. In *this project*, rename `/.devcontainer/devcontainer.wsl2.json` to `/.devcontainer/devcontainer.json`.
5. Follow [the next step in the official instructions (Open VS Code in WSL 2)](https://code.visualstudio.com/blogs/2020/07/01/containers-wsl#_open-vs-code-in-wsl-2) to open this project in a VS Code devcontainer under WSL2. (Make sure to install [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [container tools extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-containers))
6. Open *this project* in VS Code (Select File -> Open Folder). VS Code should automatically detect the devcontainer and prompt you to `Reopen in container`. If not, see [here](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) to open it manually. **Note**: Opening the container for the first time might take up to 15 min. 
7. Now you should be ready to start the the exercises.


### Manual installation

The following are the installation instructions for manually setting up the exercise code without using a container. While you can follow these instruction, you might run into installation issues and we can not promise to help you with them. (Manual installation might be required for MacOS, due to mujoco rendering from inside the container, display forwarding, and X11. See these issues: [1](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088), [2](https://github.com/google-deepmind/mujoco/issues/1047)).

1. Install a python environment manager, preferably [mamba](https://mamba.readthedocs.io/en/latest/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html), as well as [install Git](https://git-scm.com/downloads) for MacOS.
2. For conda, the following commands set up the environment 
```
conda create --name ardm -c conda-forge python=3.11
conda activate ardm
conda install pip
pip install -e .[test,cpu,pin] // for Mac: GPU is not supported
```
3. Install [acados](https://docs.acados.org/installation/).
4. Open *this project`s* code in your favorite editor. Now you should be ready to start the the exercises.

### Using CPU or GPU
By default the containers are configured to run on the CPU(, which is sufficient for the exercises). However, you can easily configure the devcontainer to run on your [cuda-enabled GPU](https://developer.nvidia.com/cuda-gpus)  for faster training of neural networks and rendering. This is especially useful for the deep reinforcement learning exercise. You require at least CUDA `>12.6.3`.

If you want to use the GPU for the exercise, first install the necessary [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) and [NVIDIA Container runtime](https://developer.nvidia.com/container-runtime) on your computer, if it is not installed already.

Then, to run the exercise container on the GPU:
1. In your `.devcontainer/devcontainer.json` uncomment the lines
```
"--gpus=all", // use only with GPU
"--runtime=nvidia" // use only with GPU
```
2. In `.devcontainer/Dockerfile` uncomment the GPU version, and comment out the CPU version
```
# FROM olivertum/arldm_student:cpu-XX
FROM olivertum/arldm_student:gpu-XX
```

3. Rebuild the VS Code Devcontainer (<kbd>ctrl+shift+p</kbd> > <kbd>Dev Containers: Rebuild and Reopen in Container</kbd>).

Executing the following in a terminal inside the container should now output `True` (if not, your GPU is not detected properly by pytorch):
```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Solving the exercises

- **Exercises** are in `src` and contain exactly one `exercise0X.ipynb` jupyter notebook. This notebook will guide you through the exercise and explains everything you need.
- **Test cases** are shipped with the exercises in `src/exercise0X/test` that allow you to test your implementations locally on your computer. You can run those tests using the [testing feature](https://code.visualstudio.com/docs/editor/testing) in VS Code, or from the command line with `pytest`, or `pytest src/exercise01`.
-  **Adhere to the following rules, otherwise the tests may fail:**
   - **Do not modify any code that you are not explicitely instructed to modify.**
   - **Do not rename files or functions.**
   - **Do not change function's arguments or return values.**
   - **Do not install any additional dependencies.**
   - :warning: **Do not post your solutions or part of the solutions publicly available during or after the course.** This also includes course-internal communication. Please strictly adhere to this rule. Not adhering to this rule will result in exemption from the course.
## Common Issues
- Common failure codes for display forwarding  include `glfw error` and `Display not found`. Try to run `xhost +local:docker` on the host. Additionally, make sure you followed all steps mentioned above.
- If building docker container fails at `RUN apt-get update`, make sure your host systems time is set correct: https://askubuntu.com/questions/1511514/docker-build-fails-at-run-apt-update-error-failed-to-solve-process-bin-sh
- `wsl --install` stuck at 0.0%: Try [this](https://github.com/microsoft/WSL/issues/9390#issuecomment-1579398805).
- If VSCode Test Discovery is stuck during test discovery without any error: [Downgrade your python extension to 2024.20.0](https://github.com/microsoft/vscode-python/issues/24656#issue-2757930549). You can also still run the test cases from a terminal in the Docker container by running the command `pytest`.

## Have fun with the exercises :partying_face:!

End of document.