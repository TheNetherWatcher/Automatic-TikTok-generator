import cv2
import json
import os
import asyncio

async def stylize(video):
    command = f"animatediff stylize create-config {video}"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        return stdout.decode()
    else:
        print(f"Error: {stderr.decode()}")

async def start_video_edit(prompt_file):
    command = f"animatediff stylize generate {prompt_file}"
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        return stdout.decode()
    else:
        print(f"Error: {stderr.decode()}")

def edit_video(video, pos_prompt):    
    x = asyncio.run(stylize(video))
    x = x.split("stylize.py")
    config = x[18].split("config =")[-1].strip()
    d = x[19].split("stylize_dir = ")[-1].strip()

    with open(config, 'r+') as f:
        data = json.load(f)
        data['head_prompt'] = pos_prompt
        data["path"] = "models/huggingface/xxmix9realistic_v40.safetensors"

    os.remove(config)
    with open(config, 'w') as f:
        json.dump(data, f, indent=4)

    out = asyncio.run(start_video_edit(d))
    out = out.split("Stylized results are output to ")[-1]
    out = out.split("stylize.py")[0].strip()

    cwd = os.getcwd()
    video_dir = cwd + "/" + out

    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
    video_path = None

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
print("The video is stored at", edit_video(video_path, pos_prompt))