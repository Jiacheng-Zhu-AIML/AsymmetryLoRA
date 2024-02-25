import sys
import numpy as np


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        

##################################
#   Code for evaluating llama-2 on MMLU benchmark
##################################

# Evaluation pipeline #1
def match(s1, s2):
    return s1.lower().strip() == s2.lower().strip()


def process_mcq(sample):
    prompt = 'Question: ' + sample['input'] + '\n'
    for c in ['A', 'B', 'C', 'D']:
        prompt += '. '.join([c, sample[c]]) + '\n'
    prompt += 'Answer:'
    return prompt, sample['target']


def measure_accuracy_by_category(model, tokenizer, text_data_dict, category_selected):
    device = "cuda"

    overall_acc_dict = {}
    overall_output_list = []
    overall_grountruth_list = []
    overall_acc_list = []
    for category in category_selected:
        # print("category =", category)
        # score_category = []
        text_data = text_data_dict[category]

        acc, o_list, t_list = measure_accuracy(model, tokenizer, text_data, category)
        
        overall_acc_dict[category] = acc
        overall_output_list.extend(o_list)
        overall_grountruth_list.extend(t_list)
        # for o, t in zip(o_list, t_list):
        #     score_category.append()
    
    overall_score = []
    for o, t in zip(overall_output_list, overall_grountruth_list):
        overall_score.append(match(o, t))
        
    overall_acc = np.mean(overall_score)
    print("overall_acc =", overall_acc)
    return overall_acc, overall_output_list ,overall_grountruth_list


def measure_accuracy_fewshot(model, tokenizer, text_data, category_selected, prompt_input_text_data=None):
    device = "cuda"
    # i = 0
    print('Running', category_selected)
    score = []
    t_list = []
    o_list = []
    
    fewshot_prompt = ""
    if prompt_input_text_data is not None:
        for s in prompt_input_text_data:
            p, t = process_mcq(s)
            fewshot_prompt += p + t + "\n\n"
    
    # print("fewshot_prompt =", fewshot_prompt)
            
    for s in text_data:
        p, t = process_mcq(s)
        
        prompt = fewshot_prompt + "\n\n" + p
        while len(tokenizer.tokenize(prompt)) + 1 > 2048:
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        o = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
        o_list.append(o)
        t_list.append(t)
        score.append(match(o, t))
    
    acc = np.mean(score)

    print('Accuracy:', acc)
    return acc, o_list, t_list


def measure_accuracy(model, tokenizer, text_data, category_selected):
    device = "cuda"
    # i = 0
    print('Running', category_selected)
    score = []
    t_list = []
    o_list = []
    for s in text_data:
        p, t = process_mcq(s)
        inputs = tokenizer(p, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        o = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)
        o_list.append(o)
        t_list.append(t)
        score.append(match(o, t))
    
    acc = np.mean(score)
    # print("o_list =", o_list)
    # print("t_list =", t_list)
    print('Accuracy:', acc)
    return acc, o_list, t_list

# Evaluation pipeline #2
def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True,))
        
        del encode_inputs
        
    answers = [answer[-1] for answer in answers]
    return answers


def process_mmlu_map(samples):
        # for sample in samples: 
        texts = [(f"{input_text},select answer from: A. {A_text}, B. {B_text}, C. {C_text}, D. {D_text} \n"
                      f"Answer:"
                      ) for input_text, A_text, B_text, C_text, D_text, answer in zip(samples['input'], 
                                                  samples['A'], samples['B'], samples['C'], samples['D'], samples['target'])]
        result = {"prompt": texts}
        return result
    

def evaluate_mmlu_acc(model, tokenizer, raw_datasets_test):
    
    model.eval()
    records = []
    
    # Repharse text as question prompt
    prompt_test = raw_datasets_test.map(
            process_mmlu_map,
            batched=True,
        )
    
    for i in range(len(raw_datasets_test)):
            prompt_text = prompt_test[i]["prompt"]
            label = prompt_test[i]["target"]
            records.append({'prompt':prompt_text, 'answer':label})
    
    pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
    gold_answers = [record['answer'] for record in records]
    run_result = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    
    print("pred_answers =", pred_answers)
    print("gold_answers =", gold_answers)
    
    count_correct = 0
    for pred, gold in zip(pred_answers, gold_answers):
        if pred == gold: 
            count_correct += 1
    acc_value = count_correct / len(pred_answers)
    
    torch.cuda.empty_cache()
    
    print("acc_value =", acc_value)


