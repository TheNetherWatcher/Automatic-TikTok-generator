import subprocess
import json
import os
import gradio as gr
import shutil

default_neg = "(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale), (multiple views), (comic), (sketch), (bad anatomy), (deformed), (disfigured), (watermark), (multiple views), (mutation hands), (mutation fingers), (extrafingers), (missing fingers), (watermark), (clothes), (covered body)"

def edit_video(video, pos_prompt, neg_prompt=None):
    print(video)
    print(pos_prompt)

    print("Video config generating")

    command = f"animatediff stylize create-config {video}"
    
    s3 = subprocess.run(command, shell=True, text=True, capture_output=True)
    # process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # stdout, stderr = process.communicate(timeout=None)
    # process.wait()

    # print("Video config generated")

    x = s3.stdout
    print(x)
    print(s3.stderr)
    x = x.split("stylize.py")

    config = x[18].split("config =")[-1].strip()
    d = x[19].split("stylize_dir = ")[-1].strip()

    with open(config, 'r+') as f:
        data = json.load(f)
        data['head_prompt'] = pos_prompt
        if neg_prompt is None:
            data['n_prompt'] = default_neg
        else:
            data['n_prompt'] = neg_prompt
        data["path"] = "share/Stable-diffusion/xxmix9realistic_v25.safetensors"

    os.remove(config)
    with open(config, 'w') as f:
        json.dump(data, f, indent=4)

    print("Prompt modified and started script")

    s = subprocess.run(f"animatediff stylize generate {d} -L 16", shell=True, capture_output=True, text=True)

    out = s.stdout
    out = out.split("Stylized results are output to ")[-1]
    out = out.split("stylize.py")[0].strip()

    cwd = os.getcwd()
    video_dir = cwd + "/" + out

    # List of common video file extensions
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}

    # Initialize variable to store the found video path
    video_path = None

    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(video_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in video_extensions:
                video_path = os.path.join(dirpath, filename)
                break
        if video_path:
            break

    return video_path

video_path = input("Enter the path to your video: ")
pos_prompt = input("Enter the what you want to do with the video: ")
neg_prompt = input(f"This is the default negative prompt: {default_neg} \n If you want to change it enter the ner negative prompt: ")

# edit_video(video_path, pos_prompt, neg_prompt)

if neg_prompt == None:
    print("The video is stored at", edit_video(video_path, pos_prompt, default_neg))
else:
    print("The video is stored at", edit_video(video_path, pos_prompt, neg_prompt))