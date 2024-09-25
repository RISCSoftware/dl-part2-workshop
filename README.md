

### Jupyter notebook

#### Option 1: Browser + Jupyter server

- Go to http://localhost:8888/tree? in your browser.

#### Option 2: VSCode + Jupyter server

- Open VSCode
- Install extensions
  - Sidebar → Extensions → \<enter name\> → Install
  - Install: Jupyter
- Connect to Jupyter Server
  - File → New File... → Jupyter Notebook
  - Ctrl+Shift+P → Notebook: Select Notebook Kernel → Existing Jupyter Server

#### Option 3: VSCode + Remote-SSH

- Open VSCode
- Install extensions
  - Sidebar → Extensions → \<enter name\> → Install
  - Install these:
    - Remote Development
- Connect via SSH
  - Ctrl+Shift+P → Remote-SSH: Connect to Host...
    - Enter `devtermgpu02`  <!-- TODO -->
  - Ctrl+Shift+P → Dev Containers: Attach to Running Container...
    - Select `dl-workshop-dev-1`
- 


### Install VSCode extensions

- ms-vscode-remote.vscode-remote-extensionpack
- ms-python.python
- lextudio.restructuredtext
- shd101wyy.markdown-preview-enhanced

### TODOs

- push container to docker hub
- send rise.css to evans

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
