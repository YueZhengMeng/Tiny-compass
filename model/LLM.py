import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseLLM:
    def __init__(self, path: str, model_name: str) -> None:
        self.path = path
        self.model_name = model_name

    def build_chat(self, tokenizer, prompt, model_name):
        pass

    def load_model_and_tokenizer(self, path, model_name, device):
        pass

    def post_process(self, response, model_name):
        pass

    def get_pred(self, data: list, max_length: int, max_gen: int, prompt_format: str, device, out_path: str):
        pass


class internlm2Chat(BaseLLM):
    def __init__(self, path: str, model_name: str = '') -> None:
        super().__init__(path, model_name)  # 调用父类初始化函数并传入参数

    def build_chat(self, prompt):
        prompt = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        return prompt

    def post_process(self, response):
        response = response.split("<|im_end|>")[0]
        return response

    def load_model_and_tokenizer(self, path, device):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
        model = model.eval()
        return model, tokenizer

    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        model, tokenizer = self.load_model_and_tokenizer(self.path, device)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True)

            prompt = self.build_chat(prompt)

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]  # 表示喂进去的tokens的长度
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=1.0,
                eos_token_id=eos_token_id,
            )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = self.post_process(pred)

            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                           "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')


class ZhipuChat(BaseLLM):
    def __init__(self, path: str = '', model_name: str = "glm-4") -> None:
        super().__init__(path, model_name)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            # 这里不方便用tokenize之后的长度，直接用字符串长度
            if len(prompt) > max_length:
                half = int(max_length / 2)
                prompt = prompt[:half] + prompt[-half:]
            # zhipu api的输入审核比较敏感，通过try except处理模型没有返回的情况
            try:
                # temperature不能等于0或者1，否则会报错
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_gen,
                    temperature=0.99,
                )
                pred = response.choices[0].message.content
            except Exception as e:
                print(e)
                pred = ""
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                           "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
