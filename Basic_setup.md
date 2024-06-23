If you're starting on a brand-new machine without any prior software installations, please follow these steps to get set up:

### 1. Install Python
1. **Download the Installer**:
   - Go to the [official Python website](https://www.python.org/downloads/).
   - Download the latest version of Python for your operating system (Windows, macOS, or Linux).

2. **Run the Installer**:
   - **Windows**:
     - Run the downloaded `.exe` file.
     - Check the box "Add Python to PATH".
     - Click "Install Now".
   - **macOS**:
     - Open the downloaded `.pkg` file.
     - Follow the on-screen instructions.
   - **Linux**:
     - Use the package manager for your distribution (e.g., `sudo apt-get install python3` for Ubuntu).

### 2. Install Anaconda
1. **Download the Installer**:
   - Go to the [Anaconda website](https://www.anaconda.com/products/distribution).
   - Download the installer for your operating system.

2. **Run the Installer**:
   - **Windows**:
     - Run the downloaded `.exe` file.
     - Follow the on-screen instructions, making sure to check the box to add Anaconda to your PATH.
   - **macOS**:
     - Open the downloaded `.pkg` file.
     - Follow the on-screen instructions.
   - **Linux**:
     - Run the following commands in your terminal:
       ```sh
       bash ~/Downloads/Anaconda3-<version>-Linux-x86_64.sh
       ```
     - Follow the on-screen instructions.

### 3. Install Jupyter Lab & Jupyter Notebook
1. **Open Anaconda Prompt/Terminal**:
   - **Windows**: Open "Anaconda Prompt".
   - **macOS/Linux**: Open your terminal.

2. **Install Jupyter Lab and Notebook**:
   ```sh
   conda install -c conda-forge jupyterlab notebook
   ```

### 4. Install Visual Studio Code or PyCharm

#### Visual Studio Code
1. **Download the Installer**:
   - Go to the [Visual Studio Code website](https://code.visualstudio.com/Download).
   - Download the installer for your operating system.

2. **Run the Installer**:
   - **Windows**:
     - Run the downloaded `.exe` file.
     - Follow the on-screen instructions.
   - **macOS**:
     - Open the downloaded `.dmg` file.
     - Drag the Visual Studio Code icon to the Applications folder.
   - **Linux**:
     - Follow the instructions specific to your distribution from the [Visual Studio Code download page](https://code.visualstudio.com/docs/setup/linux).

#### PyCharm
1. **Download the Installer**:
   - Go to the [PyCharm website](https://www.jetbrains.com/pycharm/download/).
   - Download the Community edition for your operating system.

2. **Run the Installer**:
   - **Windows**:
     - Run the downloaded `.exe` file.
     - Follow the on-screen instructions.
   - **macOS**:
     - Open the downloaded `.dmg` file.
     - Drag the PyCharm icon to the Applications folder.
   - **Linux**:
     - Follow the instructions specific to your distribution from the [PyCharm download page](https://www.jetbrains.com/pycharm/download/#section=linux).

### 5. Create an Account on GitHub
1. **Go to GitHub**:
   - Open your web browser and go to [GitHub](https://github.com/).

2. **Sign Up**:
   - Click "Sign up" in the top right corner.
   - Fill in the required details (username, email, password).
   - Follow the on-screen instructions to complete the registration process.

### Additional Setup Tips
- **Configure Git**:
  - Install Git by following instructions from the [Git website](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
  - Set up your Git username and email:
    ```sh
    git config --global user.name "Your Name"
    git config --global user.email "youremail@example.com"
    ```
- **Link Anaconda with Visual Studio Code**:
  - In Visual Studio Code, install the Python extension from the Extensions view (Ctrl+Shift+X or Cmd+Shift+X).
  - Configure Visual Studio Code to use your Anaconda Python interpreter.

By following these steps, students will be equipped with a solid foundation for their data science projects.