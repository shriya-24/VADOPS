import random
import openai
import pandas as pd
import json
import os
import re
import time

import sys
openai.api_key = "" # Use your own API Key here


# Setting hyperparameters
hyper_params = {
    "dataset" : "clinc",
    "dataset_subset" : "small"}

dataset_name = 'clinc_oos'
intent_check_list = ['whisper_mode', 'pto_balance']
dataset_subset = hyper_params['dataset_subset']
promptjson = '../prompts/templates/ChatGPT.json'
# generated_text_json = '../prompts/generated_text/ChatGPT_prompt4.json'

#temp object for examples from worst classes
worst_intent_examples = {
    "whisper_mode" : ["turn up your volume","turn your volume up"],
    "pto_balance" : ["how much paid time off have i earned to date"]
        }

def get_worst_examples(intent_class_file, intent_examples_file):
    intent_class_df = pd.read_csv(intent_class_file,sep=',')
    intent_example_df = pd.read_csv(intent_examples_file,sep=',')

    intent_class_df = intent_class_df[(intent_class_df.recall <1) & (intent_class_df.label != "oos")]
    worst_intents = intent_class_df.label.tolist()
    intent_example_df = intent_example_df[(intent_example_df['True Label']  == intent_example_df['Predicted Label']) & (intent_example_df['True Label'] != "oos")]
    worst_intent_eg = {}
    for intent in worst_intents:
        if intent not in worst_intent_eg:
            worst_intent_eg[intent] = intent_example_df[intent_example_df['True Label'] == intent].Text.tolist()
        else:
            worst_intent_eg[intent].append(intent_example_df[intent_example_df['True Label'] == intent].Text.tolist())
    return worst_intent_eg

def construct_prompt(prompttype,promptLLM,intentname,worst_intent_labels,num_eg=0,num_gen=10):
    # Opening JSON file
    promptlist = []
    with open(promptjson, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
 
    print(json_object)
    
    prompt_fill = "prompt"+str(prompttype)
    print(type(prompttype))
    if prompttype != 4:
        prompt = json_object[prompt_fill][0]

        if prompttype == 2:
            prompt = prompt.replace("{intent_name}",intentname)
            promptlist.append(prompt)
            prompt = json_object[prompt_fill][1]

    # if prompttype != 4:
            prompt = prompt.replace("{intent_name}",intentname)
    else:
        prompt = json_object[prompt_fill][intentname]
    prompt = prompt.replace("{num_gen}",str(num_gen))
    print(prompt)
    if prompttype != 4:
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
        

def get_more_data(prompttype,ic_path,ice_path,num_eg = 0,num_gen=10):

    lines_to_add = {}
    # worst_intent_data = 
    il = get_worst_examples(ic_path,ice_path)
    intent_list = list(il.keys())
    index = 0
    while index < len(intent_list):
        prompt_list = construct_prompt(prompttype,"ChatGPT",intent_list[index],il,num_eg,num_gen)
        try:
            for idx,prompt in enumerate(prompt_list):
                if idx == len(prompt_list)-1:
                    break

                #generate more questions from chatGPT
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}])

            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_list[len(prompt_list)-1]}])
        except Exception as e:
            print('Rate Limit reached or OpenAI server overloaded. so sleeping the function for 60 seconds', e)
            time.sleep(60)
            continue

        #lines to add to dataset train
        lines = []
        for l in completion.choices[0].message.content.splitlines():
            l = l.strip()
            if re.match('^\d', l):
                l = re.sub(r'^\d+\.\s+', '', l)
                lines.append(l)

        lines_to_add[intent_list[index]] = lines
        index += 1
        
    return lines_to_add

def convert_to_csv(res,generated_csv_path):
    idx = 0;
    d = pd.DataFrame()

    for i in res.keys():
        for j,sentence in enumerate(res[i]):
            temp = pd.DataFrame({
                'Sentence': sentence,
                'Label': i
            },index=[idx])

            d = pd.concat([d, temp])
            idx += 1
    # d = pd.DataFrame(
    #     [p, p.team, p.passing_att, p.passer_rating()] for p in game.players.passing()
    # )
    d.to_csv(generated_csv_path, index = False)
    return
    
if __name__ == "__main__":
#    print(os.path.abspath(os.getcwd()))
#    print(os.path.dirname(os.path.abspath(__file__)))
 
   prompt_llm = sys.argv[1]
   prompt_type = int(sys.argv[2])
   num_gen = int(sys.argv[3])

   generated_json_path = f'../prompts/generated_text/{prompt_llm}_prompt{prompt_type}.json'
   generated_csv_path = f'../prompts/generated_text/{prompt_llm}_prompt{prompt_type}.csv'
   
   ic_path = "../analysis/IntentClass_Analysis_Trainset-clinc_plus_train.csv"
   ice_path = "../analysis/Cross_entropy_analysis_train_set-clinc_plus_train.csv"
   get_worst_examples("../analysis/IntentClass_Analysis_Trainset-clinc_plus_train.csv","../analysis/Cross_entropy_analysis_train_set-clinc_plus_train.csv") 
   # return
   res = get_more_data(prompt_type,ic_path,ice_path,num_gen=num_gen)
   with open(generated_json_path,"w") as outfile:
       json.dump(res,outfile)
    
   
#    with open(generated_text_json, 'r') as openfile:
#         # Reading from json file
       # res = json.load(outfile)
 
   print(res)
   convert_to_csv(res,generated_csv_path)
   print(res)
