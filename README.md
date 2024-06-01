# Video2Video Generation using Text Prompts

This repository contains a pipeline for video-to-video generation using text prompts. The system leverages AnimateDiff and OpenPose ControlNet for pose estimation, and incorporates a prompt traveling method for improved coherence between the original and generated videos. Users can interact with this pipeline through a Gradio app or a standard Python program.

## Features

- **AnimateDiff**: Utilized for generating high-quality animations based on text prompts and an image as an input.
- **OpenPose ControlNet**: Used for accurate pose estimation to guide the animation process.
- **Prompt Traveling Method**: Ensures better relativeness and coherence between the input video and the generated output.
- **User Interfaces**: 
  - **Gradio App**: An intuitive web-based interface for easy interaction.
  - **Python Program**: A script-based interface for users preferring command-line interaction.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/TheNetherWatcher/Vid2Vid-using-Text-prompt.git
    cd Vid2Vid-using-Text-prompt
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -e .
    pip install -e .[stylize]
    ```

## Usage

### Gradio App

To run the Gradio app, execute the following command:

```bash
python app.py
```

The gradio app provides a interface for uploading video and providing a text prompt as a input and outputs the generated video

### Commandline 

```bash
python test.py
```

After running this, you will be prompted to enter the location of the video, positive prompt( the changes that you want to make in the video), and a negative prompt.
Negative is set to a default value, but you can edit it if you like.
