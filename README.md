---
title: Vid2Vid-using-Text-prompt
app_file: app.py
sdk: gradio
sdk_version: 3.35.2
---
# Automatic TikTok generator

This repository contains a pipeline for video-to-video generation using text prompts. The system leverages AnimateDiff and OpenPose ControlNet for pose estimation, and incorporates a prompt traveling method for improved coherence between the original and generated videos. Users can interact with this pipeline through a Gradio app or a standard Python program.

## Techniques used

- **AnimateDiff**: Utilized for generating high-quality animations based on text prompts and an image as an input.
- **OpenPose ControlNet**: Used for accurate pose estimation to guide the animation process.
- **Prompt Traveling Method**: Ensures better relativeness and coherence between the input video and the generated output.
- **User Interfaces**: 
  - **Gradio App**: An intuitive web-based interface for easy interaction.
  - **Python Program**: A script-based interface for users preferring command-line interaction.
 
### Base models 

- [XXMix_9realistic](https://civitai.com/models/47274): Model used for generating life-like video (Recommended for life-like video)
- [Mistoon_Anime](https://civitai.com/models/24149/mistoonanime): Model used for generating anime-like video (Recommended for anime-like video)

### Motion modules 

- [mm_sd_v15_v2](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt): Motion module used for generating segments of the final from the generated images (Recommended)
- [mm_sd_v15](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt) and [mm_sd_v14](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v14.ckpt) are some other modules that can be also used.

### ControlNets 

- [control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_openpose.pth): ControlNet for pose estimation from the given video
- Upcoming support for depth and canny controlnets too for better generated video quality.

### Prompt Travelling

This is a technique that is used to give the model, instruction at which frame what to do with the output image.
For example, if in the prompt body it is written like, 30 - face: up, camera: zoomed out, right-hand: waving, then in the output 30th frame, the image will be generated according to the given prompt.

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

### Model weights

- Download the model weights from the abve links or another, and put them [here](./data/models/huggingface), and for the downloaded motion modules, put them [here](data/models/motion-module)
- For the first time, you might get errors like model weights not found, just go to stylize directory and in the most recently created folder, edit the model name in the prompt.json file. Support for this is also under development.

### Gradio App

To run the Gradio app, execute the following command:

```bash
python app.py
```

The gradio app provides a interface for uploading video and providing a text prompt as a input and outputs the generated video.

### Commandline 

```bash
python test.py
```

After running this, you will be prompted to enter the location of the video, positive prompt (the changes that you want to make in the video), and a negative prompt.
Negative prompt is set to a default value, but you can edit it if you like.

## Upcoming Dedvelopments

- LoRA support, and controlnet(like canny, depth, edge) support
- Gradio app support for using different controlnets and LoRAs
- CLI options for controlling the execution in different system

## Credits

- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Prompt Travelling using AnimateDiff](https://github.com/s9roll7/animatediff-cli-prompt-travel)
