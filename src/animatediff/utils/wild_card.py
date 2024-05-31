import glob
import os
import random
import re

wild_card_regex = r'(\A|\W)__([\w-]+)__(\W|\Z)'


def create_wild_card_map(wild_card_dir):
    result = {}
    if os.path.isdir(wild_card_dir):
        txt_list = glob.glob( os.path.join(wild_card_dir ,"**/*.txt"), recursive=True)
        for txt in txt_list:
            basename_without_ext = os.path.splitext(os.path.basename(txt))[0]
            with open(txt, encoding='utf-8') as f:
                try:
                    result[basename_without_ext] = [s.rstrip() for s in f.readlines()]
                except Exception as e:
                    print(e)
                    print("can not read ", txt)
    return result

def replace_wild_card_token(match_obj, wild_card_map):
    m1 = match_obj.group(1)
    m3 = match_obj.group(3)

    dict_name = match_obj.group(2)

    if dict_name in wild_card_map:
        token_list = wild_card_map[dict_name]
        token = token_list[random.randint(0,len(token_list)-1)]
        return m1+token+m3
    else:
        return match_obj.group(0)

def replace_wild_card(prompt, wild_card_dir):
    wild_card_map = create_wild_card_map(wild_card_dir)
    prompt = re.sub(wild_card_regex, lambda x: replace_wild_card_token(x, wild_card_map ), prompt)
    return prompt
