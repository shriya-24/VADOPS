import random
import openai
import pandas as pd
import json
import os
import re
openai.api_key = "" # Use your own API Key here


# Setting hyperparameters
hyper_params = {
    "dataset" : "clinc",
    "dataset_subset" : "small"}

dataset_name = 'clinc_oos'
intent_check_list = ['whisper_mode', 'pto_balance']
dataset_subset = hyper_params['dataset_subset']
promptjson = './prompts/templates/ChatGPT.json'
generated_text_json = './prompts/generated_text/ChatGPT.json'

#temp object for examples from worst classes
worst_intent_examples = {
    "whisper_mode" : ["turn up your volume","turn your volume up"],
    "pto_balance" : ["how much paid time off have i earned to date"]
        }

def construct_prompt(prompttype,promptLLM,intentname,num_eg=0,num_gen=10):
    # Opening JSON file
    promptlist = []
    with open(promptjson, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
 
    print(json_object)
    
    prompt_fill = "prompt"+str(prompttype)
    prompt = json_object[prompt_fill][0]
    
    if prompttype == 2:
        prompt = prompt.replace("{intent_name}",intentname)
        promptlist.append(prompt)
        prompt = json_object[prompt_fill][1]

    prompt = prompt.replace("{intent_name}",intentname)
    prompt = prompt.replace("{num_gen}",str(num_gen))
    print(prompt)
    if num_eg > 0:
        #TODO: Add a file reading functionality to fetch examples from highest cross entropy classes
        examples += "\nExamples:\n"
        for idx,eg in enumerate(worst_intent_labels[intentname]):
            if idx+1 > num_eg:
                break
            prompt += f"{idx+1}) {eg}\n"
        prompt = prompt.replace("{examples}",examples)
    else:
        prompt = prompt.replace("{examples}","")
    promptlist.append(prompt)
    return promptlist
        

def get_more_data(prompttype,num_eg = 0,num_gen=10):

    lines_to_add = {}

    for intent_check in intent_check_list:

        prompt_list = construct_prompt(prompttype,"ChatGPT",intent_check,num_eg,num_gen)

        for idx,prompt in enumerate(prompt_list):
            if idx == len(prompt_list)-1:
                break
            
            #generate more questions from chatGPT
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}])
        print(len(prompt_list))
        completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt_list[len(prompt_list)-1]}])
        print(completion.choices[0].message.content)

        #lines to add to dataset train
        lines = []
        for l in completion.choices[0].message.content.splitlines():
            lines.append(l.split(". ")[1])

        lines_to_add[intent_check] = lines


    return lines_to_add


if __name__ == "__main__":
   print(os.path.abspath(os.getcwd()))
   print(os.path.dirname(os.path.abspath(__file__)))
   res = get_more_data(1,num_gen=20)
   with open(generated_text_json,"w") as outfile:
       json.dump(res,outfile)
   print(res)
