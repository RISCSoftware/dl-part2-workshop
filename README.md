

### Jupyter notebook

#### Option 1: Browser + Jupyter server

- Go to http://localhost:8888/tree? in your browser.

#### Option 2: VSCode + Remote-SSH

- Open VSCode
- Install extensions
  - Sidebar → Extensions → \<enter name\> → Install
  - Install these:
    - Remote Development
- Connect via SSH
  - Ctrl+Shift+P → Remote-SSH: Connect to Host...
    - Enter `qftquad2.risc.jku.at`
  - Ctrl+Shift+P → Dev Containers: Attach to Running Container...
    - Select `dl-workshop-runtime-<id>`
- 

#### Option 3: VSCode + Jupyter server

- Open VSCode
- Install extensions
  - Sidebar → Extensions → \<enter name\> → Install
  - Install: Jupyter
- Connect to Jupyter Server
  - File → New File... → Jupyter Notebook
  - Ctrl+Shift+P → Notebook: Select Notebook Kernel → Existing Jupyter Server → http://qftquad2.risc.jku.at:8888

### Install VSCode extensions

- ms-vscode-remote.vscode-remote-extensionpack
- ms-python.python
- lextudio.restructuredtext
- shd101wyy.markdown-preview-enhanced

### TODOs

- 


### Fix rise.css

- Option 1:
  ```
  .jp-Editor .cm-content {
      font-size: 150%;
  }
  ````

- Option 2:
  ```
  .cm-editor {
    font-size: 20px;
  }
  ```
