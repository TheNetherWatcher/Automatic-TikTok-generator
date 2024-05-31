import subprocess
import json
import os
import gradio as gr
import shutil

def edit_video(video, pos_prompt, neg_prompt=None):
    print(video)
    print(pos_prompt)

    save_directory = "uploads/"  # Replace with your desired folder path
    os.makedirs(save_directory, exist_ok=True)  # Create the folder if it doesn't exist

    # Generate a unique filename to prevent overwriting
    video_filename = video.split("/")[-1]  # Extract filename from path
    unique_filename = f"{save_directory}/{video_filename}"

    # Copy the uploaded video to the desired location
    shutil.copy(video, unique_filename)
    absolute_saved_path = os.path.abspath(unique_filename)
    print(absolute_saved_path)
    # copy_video = subprocess.run(f"mv {video} ")

    print("Video config generating")
    s3 = subprocess.Popen(f"animatediff stylize create-config {absolute_saved_path}", shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    s3.wait()
    out, err = s3.communicate()

    print("Video config generated")
    print("stdout:", out)
    print("stderr:", err)

    x = out  # Assign the output to x before printing it
    print(x)
    x = x.split("stylize.py")
    if len(x) < 20:  # Check if the split operation produced enough elements
        raise ValueError("Unexpected output format from subprocess: 'stylize.py' split resulted in too few elements.")

    config = x[18].split("config =")[-1].strip()
    d = x[19].split("stylize_dir = ")[-1].strip()

    with open(config, 'r+') as f:
        data = json.load(f)
        data['head_prompt'] = pos_prompt
        if neg_prompt is None:
            data['n_prompt'] = "(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale), (multiple views), (comic), (sketch), (bad anatomy), (deformed), (disfigured), (watermark), (multiple views), (mutation hands), (mutation fingers), (extrafingers), (missing fingers), (watermark), (clothes), (covered body)"
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

    os.remove(unique_filename)
    print("video found")
    return video_path

with gr.Blocks() as interface:
    gr.Markdown("## Video Processor with Text Prompts")
    with gr.Row():
        with gr.Column():
            positive_prompt = gr.Textbox(label="Positive Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")
            video_input = gr.Video(label="Input Video")
        with gr.Column():
            video_output = gr.Video(label="Processed Video")

    process_button = gr.Button("Process Video")
    process_button.click(fn=edit_video, 
                        inputs=[video_input, positive_prompt, negative_prompt],  # Add negative_prompt here
                        outputs=video_output)

interface.launch(share=True)
