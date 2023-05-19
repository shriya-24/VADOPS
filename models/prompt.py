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
no_enumerate_prompt = "\nEach generated sentence should be on a separate line, and should not be enumerated or put in bullet points."
#temp object for examples from worst classes
worst_intent_examples = {
    "whisper_mode" : ["turn up your volume","turn your volume up"],
    "pto_balance" : ["how much paid time off have i earned to date"]
        }

def get_worst_examples(intent_class_file, intent_examples_file,num_good,num_bad):
    random.seed(42)
    intent_class_df = pd.read_csv(intent_class_file,sep=',')
    intent_example_df = pd.read_csv(intent_examples_file,sep=',')

    # Currently doing only for worst performing classes
    intent_class_df = intent_class_df[(intent_class_df.recall <1) & (intent_class_df.label != "oos")]
    
    # Currently for worst intent examples
    intent_example_df_bad = intent_example_df[(intent_example_df['True Label']  != intent_example_df['Predicted Label']) & (intent_example_df['True Label'] != "oos")]
    worst_intent_eg = {}
    for intent in worst_intents:
        # TODO: decide whether num_bad is greater than the number of generated examples
        # print(intent_example_df_bad[intent_example_df_bad['True Label'] == intent].Text.tolist())
        selected_examples = random.sample(intent_example_df_bad[intent_example_df_bad['True Label'] == intent].Text.tolist(),num_bad)
        if intent not in worst_intent_eg:
            worst_intent_eg[intent] = selected_examples
        else:
            worst_intent_eg[intent] = worst_intent_eg[intent] + selected_examples
            
    # Currently for good examples
    intent_example_df_good = intent_example_df[(intent_example_df['True Label']  == intent_example_df['Predicted Label']) & (intent_example_df['True Label'] != "oos")]
    # best_intent_eg = {}
    for intent in worst_intents:
        # TODO: decide whether num_good is greater than the number of generated examples
        print(intent_example_df_good[intent_example_df_good['True Label'] == intent].Text.tolist())
        selected_examples = random.sample(intent_example_df_good[intent_example_df_good['True Label'] == intent].Text.tolist(),num_good)
        if intent not in worst_intent_eg:
            worst_intent_eg[intent] = selected_examples
        else:
            worst_intent_eg[intent] = worst_intent_eg[intent] + selected_examples
    print(worst_intent_eg)
    return worst_intent_eg 

def construct_prompt(prompttype,promptLLM,intentname,worst_intent_labels,num_eg=0,num_gen=10):
    # Opening JSON file
    promptlist = []
    with open(promptjson, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    
    prompt_fill = "prompt"+str(prompttype)
    if prompttype != 4:
        prompt = json_object[prompt_fill][0]

        if prompttype == 3:
            prompt = prompt.replace("{intent_name}",intentname)
            promptlist.append(prompt)
            prompt = json_object[prompt_fill][1]

    if prompttype != 4:
            prompt = prompt.replace("{intent_name}",intentname)
    else:
        prompt = json_object[prompt_fill][intentname]
    prompt = prompt.replace("{num_gen}",str(num_gen))
    if prompttype != 4:
        examples = ""
        if num_eg > 0:
            #TODO: Add a file reading functionality to fetch examples from highest cross entropy classes
            examples += "\nExamples:\n"
            for idx,eg in enumerate(worst_intent_labels[intentname]):
                if idx+1 > num_eg:
                    break
                examples += f"{eg}\n"
            prompt = prompt.replace("{examples}",examples)
        else:
            prompt = prompt.replace("{examples}","")
    promptlist.append(prompt)
    return promptlist
        

def get_more_data(prompttype,ic_path,ice_path,num_good,num_bad,num_eg = 0,num_gen=10):

    lines_to_add = {}
    # worst_intent_data = 
    il = get_worst_examples(ic_path,ice_path,num_good,num_bad)
    intent_list = list(il.keys())
    print(intent_list)
    return lines_to_add
    index = 0
    while index < len(intent_list):
        print(intent_list[index])
        prompt_list = construct_prompt(prompttype,"ChatGPT",intent_list[index],il,num_eg,num_gen)
        try:
            for idx,prompt in enumerate(prompt_list):
                if idx == len(prompt_list)-1:
                    break

                #generate more questions from chatGPT
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}])

            print(prompt_list)
            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt_list[len(prompt_list)-1]}])
            
        except Exception as e:
            print('Rate Limit reached or OpenAI server overloaded. so sleeping the function for 60 seconds', e)
            time.sleep(60)
            continue

        #lines to add to dataset train
        lines = []
        print(completion.choices[0].message.content.splitlines())
        for l in completion.choices[0].message.content.splitlines():
            l = l.strip()
            if re.match('^\d', l):
                l = re.sub(r'^\d+\.\s+', '', l)
                lines.append(l)

        print(lines)
        lines_to_add[intent_list[index]] = lines
        print(f"Prompt_{index} completed")
        index += 1

    return lines_to_add

def save_generated_examples(res,generated_csv_path,generated_json_path):
    with open(generated_json_path,"w") as outfile:
       json.dump(res,outfile)
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
   """
   Args:
   1. prompt_llm: Name of the Prompting LLM to be used. Useful for naming
                  the generated files
   2. prompt_type: Different type of prompting format, refer to 
                   prompts/templates/README.md for all the prompt formats. 
                   This parameter expects an integer between 1 and 4 (both 
                   inclusive)
   3. num_gen: Integer Parameter to specify how many examples to be generated from each                prompt.
        
   """
   prompt_llm = sys.argv[1]
   prompt_type = int(sys.argv[2])
   num_eg = int(sys.argv[3])
   num_good = int(sys.argv[4])
   num_bad = int(sys.argv[5])
   num_gen = int(sys.argv[6])

   generated_json_path = f'../prompts/generated_text/{prompt_llm}_prompt{prompt_type}.json'
   generated_csv_path = f'../prompts/generated_text/{prompt_llm}_prompt{prompt_type}.csv'
   
   ic_path = "../analysis/IntentClass_Analysis_Trainset-clinc_plus_train.csv"
   ice_path = "../analysis/Cross_entropy_analysis_train_set-clinc_plus_train.csv"
   # get_worst_examples("../analysis/IntentClass_Analysis_Trainset-clinc_plus_train.csv","../analysis/Cross_entropy_analysis_train_set-clinc_plus_train.csv") 
   # return
   res = get_more_data(prompt_type,ic_path,ice_path,num_good,num_bad,num_eg=num_eg,num_gen=num_gen)
    
   
#    with open(generated_text_json, 'r') as openfile:
#         # Reading from json file
       # res = json.load(outfile)
 
   save_generated_examples(res,generated_csv_path,generated_json_path)
