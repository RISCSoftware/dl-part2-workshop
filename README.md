

## Participate in Live-Coding

Welcome to the workshop!
We're excited to have you join us for this interactive learning experience.
To help you get started smoothly, here's everything you need to know about setting up your environment.

At the start of the workshop, you will receive a unique ID between 1 and 30.
In the following instructions, `<id>` will represent your specific ID.
You can already test your access by using any ID.
Feel free to run code or modify files—everything will be reset before the workshop begins.
Please note that there might be some temporary issues with VSCode.
If you encounter an error message stating that VSCode cannot execute Docker commands, please try again later.

### Option 1: Browser

- Open this link in your browser:  
  http://qftquad2.risc.jku.at:8888/tree?token=224c2703181e050254b5
- Navigate to \<id\> → workshop → 1a_torch_tensors.ipynb

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
    <!-- - Foo  
      (`lextudio.restructuredtext`)
    - Foo  
      (`shd101wyy.markdown-preview-enhanced`) -->
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
