### Example

- region prompt(txt2img / no controlnet)
- region 0 ... 1girl, upper body etc
- region 1 ... ((car)), street, road,no human etc
- background ... town, outdoors etc
- ip adapter input for background / region 0 / region 1
<img src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/ca355f4b-f4c0-4405-88f4-1c80632e32a6" width="512">

- animatediff generate -c config/prompts/region_txt2img.json -W 512 -H 768 -L 32 -C 16
- region 0 mask / region 1 mask / txt2img

<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/88eb1572-2772-4d76-89c1-c6bf8142283d" muted="false"></video></div>



<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/55042dff-4e23-43b9-b4d6-6f2228b943d2" muted="false"></video></div>
<br>

- apply different lora for each region.
- [abdiel](https://civitai.com/models/159943/abdiel-shin-megami-tensei-v-v) for region 0
- [amanozoko](https://civitai.com/models/159933/amanozoko-shin-megami-tensei-v-v) for region 1
- no lora for background
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/a5260cf0-1f96-4c65-8408-805428d2528e" muted="false"></video></div>

```json
  # new lora_map format
  "lora_map": {
        # Specify lora as a path relative to /animatediff-cli/data
        "share/Lora/zs_Abdiel.safetensors": {   # setting for abdiel lora
            "region" : ["0"],            # target region. Multiple designations possible
            "scale" : {
                # "frame_no" : scale format
                "0": 0.75           # lora scale. same as prompt_map format. For example, it is possible to set the lora to be used from the 30th frame.
            }
        },
        "share/Lora/zs_Amanazoko.safetensors": {  # setting for amanozako lora
            "region" : ["1"],            # target region
            "scale" : {
                "0": 0.75
            }
        }
  },
```
- more example [here](https://github.com/s9roll7/animatediff-cli-prompt-travel/issues/147)
<br>



- img2img
- This can be improved using controlnet, but this sample does not use it.
- source / denoising_strength 0.7 / denoising_strength 0.85
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/4a9ecf3e-d1c7-468a-a85f-abce7b4c4aab" muted="false"></video></div>
<br>
<br>

- [A command to stylization with region has been added](https://github.com/s9roll7/animatediff-cli-prompt-travel#video-stylization-with-region).
- (You can also create json manually without using the stylize command.)
- region prompt
- Region division into person shapes
- source / img2img / txt2img
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/10101ab8-39cc-4dbc-85c8-af558f80c6fe" muted="false"></video></div>
<br>

- source / Region division into person shapes / inpaint
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/d9231c8e-94b6-4608-97ba-6ab3fc85bcfa" muted="false"></video></div>
<br>
<br>




- [A command to stylization with mask has been added](https://github.com/s9roll7/animatediff-cli-prompt-travel#video-stylization-with-mask).
- more example [here](https://github.com/s9roll7/animatediff-cli-prompt-travel/issues/111)

<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/461cd68a-511a-4ad8-a2ee-7078faed7354" muted="false"></video></div>
<br>


- [A command to automate video stylization has been added](https://github.com/s9roll7/animatediff-cli-prompt-travel#video-stylization).
- Original / First generation result / Second generation(for upscaling) result
- It took 4 minutes to generate the first one and about 5 minutes to generate the second one (on rtx 4090).
- more example [here](https://github.com/s9roll7/animatediff-cli-prompt-travel/issues/29)

<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/2f1965f2-9a50-485e-ac95-e888a3189ba2" muted="false"></video></div>
<br>


- controlnet_openpose + controlnet_softedge
- input frames for controlnet(0,16,32 frames)
<img src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/4adac698-75a4-4c6d-bf64-a5723d0e3e77" width="512">

- result
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/50aa9d0d-15b6-4c84-a497-8d020d3bdb7c" muted="false"></video></div>
<br>

- In the latest version, generation can now be controlled more precisely through prompts.
- sample 1
```json
    "prompt_fixed_ratio": 0.8,
    "head_prompt": "1girl, wizard, circlet, earrings, jewelry, purple hair,",
    "prompt_map": {
        "0": "(standing,full_body),blue_sky, town",
        "8": "(sitting,full_body),rain, town",
        "16": "(standing,full_body),blue_sky, woods",
        "24": "(upper_body), beach",
        "32": "(upper_body, smile)",
        "40": "(upper_body, angry)",
        "48": "(upper_body, smile, from_above)",
        "56": "(upper_body, angry, from_side)",
        "64": "(upper_body, smile, from_below)",
        "72": "(upper_body, angry, from_behind, looking at viewer)",
        "80": "face,looking at viewer",
        "88": "face,looking at viewer, closed_eyes",
        "96": "face,looking at viewer, open eyes, open_mouth",
        "104": "face,looking at viewer, closed_eyes, closed_mouth",
        "112": "face,looking at viewer, open eyes,eyes, open_mouth, tongue, smile, laughing",
        "120": "face,looking at viewer, eating, bowl,chopsticks,holding,food"
    },
```
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/c4de4b87-f302-4d61-98c7-9607dece386f" muted="false"></video></div>
<br>

- sample 2
```json
    "prompt_fixed_ratio": 1.0,
    "head_prompt": "1girl, wizard, circlet, earrings, jewelry, purple hair,",
    "prompt_map": {
        "0": "",
        "8": "((fire magic spell, fire background))",
        "16": "((ice magic spell, ice background))",
        "24": "((thunder magic spell, thunder background))",
        "32": "((skull magic spell, skull background))",
        "40": "((wind magic spell, wind background))",
        "48": "((stone magic spell, stone background))",
        "56": "((holy magic spell, holy background))",
        "64": "((star magic spell, star background))",
        "72": "((plant magic spell, plant background))",
        "80": "((meteor magic spell, meteor background))"
    },
```
<div><video controls src="https://github.com/s9roll7/animatediff-cli-prompt-travel/assets/118420657/31a5827d-e551-4937-8b67-51747a92d14c" muted="false"></video></div>
<br>

