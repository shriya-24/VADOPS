from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys


def save_prompt_results_flan_xl(f,g):
    # Currently using only FLAN-t5-XL
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    prompt_list = f.readlines()


    for l in prompt_list:
        print(l)
        inputs = tokenizer(l, return_tensors="pt")
        outputs = model.generate(**inputs, do_sample=True, min_length=100, max_length=700, temperature=0.97, repetition_penalty=1.2)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        g.write(result[0])


if __name__ == "__main__":
    arg_list = sys.argv
    prompt_file = arg_list[1]
    prompt_result_file = arg_list[2]
    print(prompt_result_file)
    
    f = open(prompt_file, "r")
    g = open(prompt_result_file,"w")
    
    save_prompt_results_flan_xl(f,g)
    
    f.close()
    g.close()