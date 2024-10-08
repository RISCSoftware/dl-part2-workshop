

## Abstract

**Advanced Machine Learning Workshop: From Theory to Practice with Neural Networks**

We cordially invite you to participate in a comprehensive workshop designed to enrich your understanding of machine learning.
We will explore the theoretical foundations and practical applications of neural networks, emphasizing the transformative technologies of transformers and attention mechanisms.

Throughout this three-hour workshop, we will explore both theory and hands-on coding, using Python and Jupyter notebooks to apply learning in real-time.
We will cover essential aspects such as neural network architecture, emphasizing initialization, regularization, optimization, backpropagation, and first-order automatic gradient differentiation.
These concepts form the backbone of advanced machine learning practices.
In our session on natural language processing (NLP), we will discuss and code the attention mechanism and transformer architectures, combining theoretical insights with practical coding exercises.
The workshop will also include discussions and examples on select papers in the field of neurosymbolic AI, exploring current trends and innovative methodologies.

Please bring your laptop to fully participate, as each session integrates coding activities where we implement and train key components of neural networks using PyTorch.
Detailed instructions for preparing your laptop, including necessary installations and configurations, will be provided soon.
We give our best to offer a workshop that will enhance your understanding of neural networks and provide practical experience with advanced machine learning techniques for tackling complex challenges.

## How to run the workshop code?

<!-- ### Clone the repo and its submodules

```
git clone https://github.com/RISCSoftware/dl-part2-workshop
git submodule update --init --recursive
``` -->

### Option 1: Run a docker container

- Install docker
- Start our docker container
  ```
  docker run -p 8888:8888 risclidse/dl-workshop bash -c "jupyter notebook --NotebookApp.ip=0.0.0.0 --NotebookApp.port=8888 --NotebookApp.allow_root=True --NotebookApp.notebook_dir=/repo --NotebookApp.token=''"
  ```
- Enter the provided URL in the browser, e.g. http://127.0.0.1:8888/tree

### Option 2: Run the notebooks on Google Colab

- Download the `.ipynb` files of our workshop
- Go to https://colab.research.google.com/
- Select File → Open notebook → Upload
- Note:
  - You might be asked to restart the session once after the `pip3 install` completed
  - Most images will not be shown on Colab

## Participate in Live-Coding

Welcome to the workshop!
We're excited to have you join us for this interactive learning experience.
To help you get started smoothly, here's everything you need to know about setting up your environment.

At the start of the workshop, you will receive a unique ID between 1 and 30.
In the following instructions, `<id>` will represent your specific ID.
You can already test your access by using any ID.
Feel free to run code or modify files—everything will be reset before the workshop begins.

Please be aware that we are currently experiencing issues with the VSCode option.
If you encounter an error message stating that VSCode cannot execute Docker commands, please try again later.

If you see only a blank page in the Browser option, try `Ctrl+Shift+R`.

### Option 1: Browser

- Open this link in your browser:  
  http://qftquad2.risc.jku.at:8888/tree?token=224c2703181e050254b5
- Navigate to \<id\> → repo → workshop → 1a_torch_tensors.ipynb

### Option 2: VSCode

- Install or update VSCode on your laptop
  - Via an executable or using a command, e.g. in Windows:
    ```
    winget install -e --id Microsoft.VisualStudioCode
    winget update -e --id Microsoft.VisualStudioCode
    ```
    In Debian Linux:
    ```
    sudo apt update
    sudo apt install code
    sudo apt upgrade code
    ```
- Open VSCode and install extensions
  - Sidebar → Extensions → \<enter name\> → Install
    - Remote Development  
      `ms-vscode-remote.vscode-remote-extensionpack`
    - Python  
      `ms-python.python`
    - Jupyter  
      `ms-toolsai.jupyter`
- Connect via SSH
  - Ctrl+Shift+P → Remote-SSH: Connect to Host...
    - Enter `<user>@qftquad2.risc.jku.at`  
      Note: Make sure to prefix your user on Windows
    - Enter your password
  - Ctrl+Shift+P → Dev Containers: Attach to Running Container...
    - Select `dl-workshop-runtime-<id>`
    - Enter your password again
- Sidebar → Explorer
- Open Filder → `/repo/`
- Navigate to `workshop/1a_torch_tensors.ipynb`

<!-- ### Option 3: VSCode + Jupyter server

- Open VSCode
- Install extensions
  - Sidebar → Extensions → \<enter name\> → Install
  - Install: Jupyter
- Connect to Jupyter Server
  - File → New File... → Jupyter Notebook
  - Ctrl+Shift+P → Notebook: Select Notebook Kernel → Existing Jupyter Server → http://qftquad2.risc.jku.at:8888 -->

## File structure

- [d2l-en/](d2l-en) - Directory containing the English version of the "Dive into Deep Learning" (D2L) resources.
- [d2l-pytorch-colab/](d2l-pytorch-colab) - Colab notebooks for D2L with PyTorch, ready to be used in Colab environments.
- [d2l-pytorch-slides/](d2l-pytorch-slides) - Slide decks for presentations based on D2L content using PyTorch.
- [images/](images) - A folder containing images used throughout the workshop materials and notebooks.
- [workshop/](workshop) - Main directory for workshop-related materials.
  - [0_intro.ipynb](workshop/0_intro.ipynb) - Introduction to the workshop, including initial setup and background information.
  - [1a_torch_tensors.ipynb](workshop/1a_torch_tensors.ipynb) - Tensor manipulation in PyTorch.
  - [1b_neural_nets.ipynb](workshop/1b_neural_nets.ipynb) - Basics of neural networks in PyTorch.
  - [2_transformers_self_attention.ipynb](workshop/2_transformer_self_attention.ipynb) - Introduction to transformers and self-attention mechanisms.
  - [3_neurosymbolic_ai.ipynb](workshop/3_neurosymbolic_ai.ipynb) - Neurosymbolic AI concepts and applications.
- [utils/](utils) - Utility scripts and helper functions.

## Attribution and Licence

This project uses content from the following sources:

- [Dive into Deep Learning (d2l)](https://github.com/d2l-ai/d2l-en) - Licensed under CC BY-SA 4.0. See the [LICENSE](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE).
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by [Jay Alammar](https://jalammar.github.io/), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- [Neuromatch Academy Course Content](https://github.com/NeuromatchAcademy/course-content-dl) - Licensed under CC BY-4.0 and BSD-3-Clause.
- [Understanding Deep Learning (udlbook)](https://github.com/udlbook/udlbook) - Licensed under CC BY-NC-ND 4.0. See the [LICENSE](https://github.com/udlbook/udlbook/blob/main/LICENSE).
- [A Survey of Planning and Learning in Games](https://www.researchgate.net/publication/342579276_A_Survey_of_Planning_and_Learning_in_Games)

Please refer to the links for specific license details.
