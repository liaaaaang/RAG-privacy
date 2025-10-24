import warnings
import json
import os
import glob
import fire
import torch
from dotenv import load_dotenv

# 尝试导入 Meta 官方实现（原始权重用）
try:
    from llama import llama as meta_llama
except Exception:
    meta_llama = None

# Hugging Face 路径（HF 转换权重用）
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

load_dotenv()


def _is_meta_original_format(model_dir: str) -> bool:
    """是否为 Meta 原始发布格式（consolidated.xx.pth + params.json）"""
    if not os.path.isdir(model_dir):
        return False
    has_params = os.path.exists(os.path.join(model_dir, "params.json"))
    has_consolidated = len(glob.glob(os.path.join(model_dir, "consolidated.*.pth"))) > 0
    return has_params and has_consolidated


def _is_hf_format(model_dir: str) -> bool:
    """是否为 Hugging Face 转换格式（config.json + pytorch_model-*.bin / *.safetensors）"""
    if not os.path.isdir(model_dir):
        return False
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_weights = (
        len(glob.glob(os.path.join(model_dir, "pytorch_model*.bin"))) > 0
        or len(glob.glob(os.path.join(model_dir, "model*.safetensors"))) > 0
    )
    return has_config and has_weights


class HFGenerator:
    """
    用 HF 本地模型模拟现有的 generator.text_completion([...]) 接口
    返回 [{'generation': str}] 的列表，与原脚本保持一致
    """
    def __init__(self, model_dir: str, max_seq_len: int = 4096, max_batch_size: int = 1):
        assert AutoTokenizer is not None and AutoModelForCausalLM is not None, \
            "请安装 transformers / safetensors：pip install transformers safetensors"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 纯离线加载
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        # 尽量自动用半精度；CPU 就退回到 float32
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype
        )
        if self.device == "cuda":
            self.model.to(self.device)
        # 生成时用
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        # 有些 tokenizer 没有 pad_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def text_completion(self, prompts, max_gen_len=256, temperature=0.6, top_p=0.9):
        # 简单串行；如需批处理可自行按 max_batch_size 分批
        outputs = []
        for prompt in prompts:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len
            ).to(self.device)
            gen_ids = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_gen_len,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # 只取新增部分（等价于你们原实现的 'generation' 字段）
            new_tokens = gen_ids[0][enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append({"generation": text})
        return outputs


def _build_generator(model_path: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int):
    """
    根据目录自动选择 Meta 原始 or HF 转换
    """
    if _is_meta_original_format(model_path):
        assert meta_llama is not None, "未找到 Meta 官方实现 llama 依赖，请检查环境。"
        return meta_llama.Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=os.path.join(os.path.dirname(model_path), tokenizer_path)
            if os.path.exists(os.path.join(os.path.dirname(model_path), tokenizer_path))
            else tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
    elif _is_hf_format(model_path):
        return HFGenerator(model_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    else:
        raise AssertionError(
            f"在 {model_path} 未识别到有效模型文件："
            f"既不是 Meta 原始（consolidated.*.pth + params.json），也不是 HF（config.json + 权重）"
        )


def main(
    ckpt_dir: str,              # 传“模型名”或“完整目录”均可
    path: str,                  # 输入/输出所在相对目录（Inputs&Outputs/{path}/...）
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 256,
    max_batch_size: int = 1,
):
    print(path)

    # 解析实际模型目录：优先拼 LLM_MODEL_PATH，再尝试 ./Model，最后当作绝对/相对路径
    base_llm_path = os.path.expanduser(os.environ.get('LLM_MODEL_PATH', '~/workspace/model/llama'))
    if os.path.exists(os.path.join(base_llm_path, ckpt_dir)):
        model_path = os.path.join(base_llm_path, ckpt_dir)
    elif os.path.exists(os.path.join('Model', ckpt_dir)):
        model_path = os.path.join('Model', ckpt_dir)
    else:
        model_path = ckpt_dir

    # 供文件命名的 tag（防止把绝对路径写进文件名里）
    ckpt_tag = os.path.basename(model_path.rstrip('/'))

    # ——（可选）摘要阶段：同样支持 HF / Meta 两种目录——
    if os.path.exists(f'./Inputs&Outputs/{path}/set.json'):
        print('summarizing now')
        with open(f'./Inputs&Outputs/{path}/set.json', "r") as file:
            settings = json.load(file)
        summary_model_name = settings['infor']
        para_flag = False
        if '-para' in summary_model_name:
            para_flag = True
            summary_model_name = summary_model_name.replace('-para', '')

        # 定位摘要模型目录
        if os.path.exists(os.path.join(base_llm_path, summary_model_name)):
            summary_model_path = os.path.join(base_llm_path, summary_model_name)
        elif os.path.exists(os.path.join('Model', summary_model_name)):
            summary_model_path = os.path.join('Model', summary_model_name)
        else:
            summary_model_path = summary_model_name

        # 自动选择加载器
        summary_generator = _build_generator(summary_model_path, tokenizer_path, max_seq_len, max_batch_size)

        suf = settings['suffix']
        adh_1 = settings['adhesive_con']
        adh_2 = settings['adhesive_prompt']
        with open(f"./Inputs&Outputs/{path}/question.json", 'r', encoding='utf-8') as f_que:
            questions = json.loads(f_que.read())
        with open(f"./Inputs&Outputs/{path}/context.json", 'r', encoding='utf-8') as f_con:
            contexts = json.loads(f_con.read())

        su_1 = ("Given the following question and context, extract any part of the context"
                " *AS IS* that is relevant to answer the question. If none of the context is relevant"
                " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Question: ")
        if para_flag:
            su_1 = ("Given the following question and context, extract any part of the context"
                    " *AS IS* that is relevant to answer the question. If none of the context is relevant"
                    " return NO_OUTPUT.\n\n> Question: ")
        su_2 = "\n> Context:\n>>>\n"
        su_3 = "\n>>>\nExtracted relevant parts:"

        prompt_ge_contexts, summarize_contexts = [], []
        for i in range(len(questions)):
            ques = questions[i]
            k_contexts = contexts[i]
            ge_contexts, sum_contexts = [], []
            for j in range(len(k_contexts)):
                context = k_contexts[j]
                prompt_ge_context = su_1 + ques + su_2 + context + su_3
                ge_contexts.append(prompt_ge_context)
                results = summary_generator.text_completion(
                    [prompt_ge_context],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                ans = results[0]['generation']
                sum_contexts.append(ans)
            summarize_contexts.append(sum_contexts)
            prompt_ge_contexts.append(ge_contexts)

        with open(f"./Inputs&Outputs/{path}/summarize_contexts.json", 'w', encoding='utf-8') as f_c:
            f_c.write(json.dumps(summarize_contexts))
        with open(f"./Inputs&Outputs/{path}/generate_summarize_prompt.json", 'w', encoding='utf-8') as f_g:
            f_g.write(json.dumps(prompt_ge_contexts))

        prompts = []
        for i in range(len(questions)):
            con_u = adh_1.join(summarize_contexts[i])
            prompt = suf[0] + con_u + adh_2 + suf[1] + questions[i] + adh_2 + suf[2]
            prompts.append(prompt)
        with open(f"./Inputs&Outputs/{path}/prompts.json", 'w', encoding='utf-8') as f_p:
            f_p.write(json.dumps(prompts))

    # ——主模型：自动选择加载器（Meta 原始 or HF 转换）——
    generator = _build_generator(model_path, tokenizer_path, max_seq_len, max_batch_size)

    # 推理
    print('generating output')
    with open(f"./Inputs&Outputs/{path}/prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.loads(f.read())

    answers = []
    for i in range(len(all_prompts)):
        out = generator.text_completion(
            [all_prompts[i]],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        answers.append(out[0]['generation'])

    # 注意：用 ckpt_tag（目录名）统一输出命名，避免绝对路径导致 evaluation 对不上
    out_path = f"./Inputs&Outputs/{path}/outputs-{ckpt_tag}-{temperature}-{top_p}-{max_seq_len}-{max_gen_len}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(answers))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fire.Fire(main)
