"""
改进版的检索数据库实现，完全使用本地嵌入模型
This file implements a retrieval database using local embedding models.
"""
import os
import random
import shutil
import json
import argparse
from typing import List, Optional, Dict, Any
from chardet.universaldetector import UniversalDetector
from pathlib import Path

import torch
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from nltk.tokenize import RegexpTokenizer
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Load environment variables
load_dotenv()

# Get data root from environment variable
DATA_ROOT = os.path.expanduser(os.environ.get('DATA_ROOT', '~/workspace/data'))
EMBEDDING_MODEL_PATH = os.path.expanduser(os.environ.get('EMBEDDING_MODEL_NAME', '~/workspace/model/embedding'))


class LocalHuggingFaceEmbeddings(Embeddings):
    """
    自定义的本地 HuggingFace 嵌入模型类
    完全使用本地模型，不需要网络连接
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        pooling_method: str = 'mean'  # 'mean' or 'cls'
    ):
        """
        初始化本地嵌入模型

        Args:
            model_path: 本地模型路径
            device: 运行设备 ('cpu' or 'cuda')
            batch_size: 批处理大小
            normalize_embeddings: 是否归一化嵌入向量
            pooling_method: 池化方法 ('mean' or 'cls')
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method

        # 加载模型和分词器
        print(f"Loading local model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(device)
        self.model.eval()

        # 获取模型维度
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

    def _mean_pooling(self, model_output, attention_mask):
        """均值池化"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _cls_pooling(self, model_output):
        """CLS token 池化"""
        return model_output[0][:, 0]

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """编码一批文本"""
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Pooling
        if self.pooling_method == 'mean':
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.pooling_method == 'cls':
            embeddings = self._cls_pooling(model_output)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Normalize embeddings
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        embeddings = self._encode_batch([text])
        return embeddings[0].tolist()


def find_all_file(path: str) -> List[str]:
    """返回文件夹中的所有文件列表"""
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_encoding_of_file(path: str) -> str:
    """返回文件的编码格式"""
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        data = file.readlines()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']


def get_local_embed_model(
    encoder_model_name: str,
    device: str = 'cpu',
    retrival_database_batch_size: int = 256,
    normalize_embeddings: bool = True,
    pooling_method: str = 'mean'
) -> LocalHuggingFaceEmbeddings:
    """
    获取本地嵌入模型

    Args:
        encoder_model_name: 模型名称或路径
        device: 运行设备
        retrival_database_batch_size: 批处理大小
        normalize_embeddings: 是否归一化
        pooling_method: 池化方法

    Returns:
        本地嵌入模型实例
    """
    # 预定义的模型路径映射
    model_path_map = {
        'all-MiniLM-L6-v2': os.path.join(EMBEDDING_MODEL_PATH, 'all-MiniLM-L6-v2'),
        'bge-large-en-v1.5': os.path.join(EMBEDDING_MODEL_PATH, 'bge-large-en-v1.5'),
        'bge-large-zh-v1.5': os.path.join(EMBEDDING_MODEL_PATH, 'bge-large-zh-v1.5'),
        'e5-base-v2': os.path.join(EMBEDDING_MODEL_PATH, 'e5-base-v2'),
        'e5-large-v2': os.path.join(EMBEDDING_MODEL_PATH, 'e5-large-v2'),
        'm3e-base': os.path.join(EMBEDDING_MODEL_PATH, 'm3e-base'),
        'm3e-large': os.path.join(EMBEDDING_MODEL_PATH, 'm3e-large'),
    }

    # 确定模型路径
    if encoder_model_name in model_path_map:
        model_path = model_path_map[encoder_model_name]
    elif os.path.exists(encoder_model_name):
        # 如果是完整路径
        model_path = encoder_model_name
    else:
        # 尝试作为子目录
        model_path = os.path.join(EMBEDDING_MODEL_PATH, encoder_model_name)

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise ValueError(f"模型路径不存在: {model_path}")

    # 检查必要的模型文件
    required_files = ['config.json', 'pytorch_model.bin']
    alternative_files = ['model.safetensors']  # 新版本可能使用 safetensors

    has_required = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
    has_alternative = any(os.path.exists(os.path.join(model_path, f)) for f in alternative_files)

    if not (has_required or has_alternative):
        raise ValueError(f"模型文件不完整，请确保 {model_path} 包含必要的模型文件")

    # 创建本地嵌入模型
    embed_model = LocalHuggingFaceEmbeddings(
        model_path=model_path,
        device=device,
        batch_size=retrival_database_batch_size,
        normalize_embeddings=normalize_embeddings,
        pooling_method=pooling_method
    )

    return embed_model


def pre_process_dataset(data_name: str, change_method: str = 'body') -> None:
    """预处理数据集用于检索数据库"""
    data_store_path = DATA_ROOT

    def pre_process_chatdoctor() -> None:
        """删除 chatdoctor 数据集中的指令"""
        file_path = os.path.join(data_store_path, 'chatdoctor200k/chatdoctor200k.json')
        if not os.path.exists(file_path):
            print(f"警告: {file_path} 不存在。请确保数据已下载。")
            return

        with open(file_path, 'r') as f:
            content = f.read()
            data = json.loads(content)

        output_dir = os.path.join(data_store_path, 'chatdoctor')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'chatdoctor.txt')

        with open(output_path, 'w', encoding="utf-8") as f:
            max_len = 0
            for i, item in enumerate(data):
                s = 'input: ' + item['input'] + '\n' + 'output: ' + item['output']
                s = s.replace('\xa0', ' ')
                if i != len(data) - 1:
                    s += '\n\n'
                max_len = max(max_len, len(s))
                f.write(s)
        print(f'chatdoctor 数据集数量: {len(data)}')
        print(f'chatdoctor 数据集最大长度: {max_len}')

    def pre_process_enron_mail() -> None:
        num_file = 0
        data_path = os.path.join(data_store_path, data_name)
        if not os.path.exists(data_path):
            print(f"警告: {data_path} 不存在。请确保数据已下载。")
            return

        for file_name in find_all_file(data_path):
            encoding = get_encoding_of_file(file_name)
            with open(file_name, 'r', encoding=encoding) as file:
                data = file.read()
            content = data.split('\n\n')
            new_content = ""
            for item in content:
                item_ = item.strip()
                if item_ == '':
                    continue
                if change_method == 'body':
                    num_other_title = 0
                    other_messages = ["Message-ID:", "Date:", "From:", "To:", "Subject:", "Mime-Version:", "X-Origin:",
                                    "Cc:", "Content-Transfer-Encoding:", "X-From:", "X-To:", "X-cc:", "X-bcc:",
                                    "Sent:", "X-Folder:", "X-FileName:", "Content-Type:", "Bcc:", "X-Origin:",
                                    "X-FileName:"]
                    for other_message in other_messages:
                        num_other_title += item_.count(other_message)
                    if num_other_title < 3:
                        new_content += item_.replace('\n', ' ')
                elif change_method == 'strip':
                    new_content += item_.replace('\n', ' ')
                new_content = new_content.strip()
                if new_content != "" and new_content[-1] not in '.?!':
                    new_content += '.'
            if len(new_content) != 0:
                relative_path = os.path.relpath(file_name, data_path)
                output_dir = os.path.join(data_store_path, f'enron-mail-{change_method}')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, relative_path + '.txt')
                output_file_dir = os.path.dirname(output_file)
                num_file += 1
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
        print(f'{data_name}-{change_method} 文件数量: {num_file}')

    if data_name == "chatdoctor200k":
        pre_process_chatdoctor()
    elif data_name == "enron-mail":
        pre_process_enron_mail()


def split_dataset(data_name: str, split_ratio: float = 0.99, num_eval: int = 1000, max_que_len: int = 50) -> None:
    """将数据集分割为训练集和测试集"""
    data_store_path = DATA_ROOT
    data_path = os.path.join(data_store_path, data_name)

    if not os.path.exists(data_path):
        print(f"错误: {data_path} 不存在")
        return

    if data_name.find('chatdoctor') != -1:
        file_path = os.path.join(data_path, 'chatdoctor.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        data = content.split('\n\n')
        random.shuffle(data)
        num_train = int(len(data) * split_ratio)
        train_data = data[:num_train]
        test_data = data[num_train:]

        # 创建输出目录
        train_dir = os.path.join(data_store_path, f'{data_name}-train')
        test_dir = os.path.join(data_store_path, f'{data_name}-test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 保存训练数据
        with open(os.path.join(train_dir, 'chatdoctor.txt'), 'w', encoding='utf-8') as file:
            for i, item in enumerate(train_data):
                file.write(item)
                if i != len(train_data) - 1:
                    file.write('\n\n')

        # 保存测试数据
        with open(os.path.join(test_dir, 'chatdoctor.txt'), 'w', encoding='utf-8') as file:
            for i, item in enumerate(test_data):
                file.write(item)
                if i != len(test_data) - 1:
                    file.write('\n\n')

        # 生成评估输入
        eval_data = test_data[:num_eval]
        eval_input = []
        tokenizer = RegexpTokenizer(r'\w+')
        for data_item in eval_data:
            que = tokenizer.tokenize(data_item)[:max_que_len]
            eval_input.append(' '.join(que))
        with open(os.path.join(test_dir, 'eval_input.json'), 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))

    elif data_name.find('enron-mail') != -1:
        all_file = list(find_all_file(data_path))
        random.shuffle(all_file)
        num_train = int(len(all_file) * split_ratio)
        train_all_file = all_file[:num_train]
        test_all_file = all_file[num_train:]

        # 创建输出目录
        train_dir = os.path.join(data_store_path, f'{data_name}-train')
        test_dir = os.path.join(data_store_path, f'{data_name}-test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for train_file in train_all_file:
            source_file = train_file
            relative_path = os.path.relpath(train_file, data_path)
            target_file = os.path.join(train_dir, relative_path)
            target_dir = os.path.dirname(target_file)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy2(source_file, target_file)

        for test_file in test_all_file:
            source_file = test_file
            relative_path = os.path.relpath(test_file, data_path)
            target_file = os.path.join(test_dir, relative_path)
            target_dir = os.path.dirname(target_file)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy2(source_file, target_file)

        # 生成评估输入
        random.shuffle(test_all_file)
        eval_data = test_all_file[:num_eval]
        eval_input = []
        tokenizer = RegexpTokenizer(r'\w+')
        for path_eval_data in eval_data:
            encoding = get_encoding_of_file(path_eval_data)
            with open(path_eval_data, 'r', encoding=encoding) as file:
                data = file.read()
            que = tokenizer.tokenize(data)[:max_que_len]
            eval_input.append(' '.join(que))
        with open(os.path.join(test_dir, 'eval_input.json'), 'w', encoding='utf-8') as file:
            file.write(json.dumps(eval_input))


class SingleFileSplitter(TextSplitter):
    """单文件分割器 - 不分割文本"""
    def split_text(self, text: str) -> List[str]:
        return [text]


class LineBreakTextSplitter(TextSplitter):
    """双换行符分割器"""
    def split_text(self, text: str) -> List[str]:
        return text.split("\n\n")


def construct_retrieval_database(
    data_name_list: List[str],
    split_method: Optional[List[str]] = None,
    encoder_model_name: str = 'all-MiniLM-L6-v2',
    retrival_database_batch_size: int = 256,
    chunk_size: int = 1500,
    chunk_overlap: int = 100,
    normalize_embeddings: bool = True,
    pooling_method: str = 'mean'
) -> Chroma:
    """
    从数据集构建检索数据库

    Args:
        data_name_list: 数据集名称列表
        split_method: 分割方法列表
        encoder_model_name: 编码器模型名称
        retrival_database_batch_size: 批处理大小
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        normalize_embeddings: 是否归一化嵌入
        pooling_method: 池化方法

    Returns:
        Chroma 向量数据库实例
    """

    def get_splitter(split_method_):
        """根据方法名获取分割器"""
        if split_method_ == 'single_file':
            return SingleFileSplitter()
        elif split_method_ == 'by_two_line_breaks':
            return LineBreakTextSplitter()
        elif split_method_ == 'recursive_character':
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"未知的分割方法: {split_method_}")

    data_store_path = DATA_ROOT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 处理分割方法
    if split_method is None:
        split_method = ['single_file'] * len(data_name_list)
    elif len(split_method) == 1:
        split_method = split_method * len(data_name_list)
    else:
        assert len(split_method) == len(data_name_list)

    split_texts = []
    for n_data_name, data_name in enumerate(data_name_list):
        documents = []
        # 打开文件
        data_path = os.path.join(data_store_path, data_name)
        if not os.path.exists(data_path):
            print(f"警告: {data_path} 不存在。跳过...")
            continue

        for file_name in find_all_file(data_path):
            # 检测文件编码
            encoding = get_encoding_of_file(file_name)
            # 加载数据
            loader = TextLoader(file_name, encoding=encoding)
            doc = loader.load()
            documents.extend(doc)

        print(f'{data_name} 的文件数量: {len(documents)}')
        # 获取分割器
        splitter = get_splitter(split_method[n_data_name])
        # 分割文本
        split_texts += splitter.split_documents(documents)

    # 获取本地嵌入模型
    embed_model = get_local_embed_model(
        encoder_model_name,
        device,
        retrival_database_batch_size,
        normalize_embeddings,
        pooling_method
    )

    # 构建数据库名称
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name

    # 向量存储路径
    vector_store_path = f"./RetrievalBase/{retrieval_name}/{encoder_model_name}"
    print(f'生成 {retrieval_name} 的 Chroma 数据库，使用模型 {encoder_model_name}')

    # 创建向量数据库
    retrieval_database = Chroma.from_documents(
        documents=split_texts,
        embedding=embed_model,
        persist_directory=vector_store_path
    )

    print(f"数据库创建完成，包含 {len(split_texts)} 个文档片段")
    print(f"数据库保存在: {vector_store_path}")

    return retrieval_database


def load_retrieval_database_from_address(
    store_path: str,
    encoder_model_name: str = 'all-MiniLM-L6-v2',
    retrival_database_batch_size: int = 512,
    normalize_embeddings: bool = True,
    pooling_method: str = 'mean'
) -> Chroma:
    """
    从地址加载预构建的检索数据库

    Args:
        store_path: 数据库存储路径
        encoder_model_name: 编码器模型名称
        retrival_database_batch_size: 批处理大小
        normalize_embeddings: 是否归一化嵌入
        pooling_method: 池化方法

    Returns:
        Chroma 向量数据库实例
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_local_embed_model(
        encoder_model_name,
        device,
        retrival_database_batch_size,
        normalize_embeddings,
        pooling_method
    )
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


def load_retrieval_database_from_parameter(
    data_name_list: List[str],
    encoder_model_name: str = 'all-MiniLM-L6-v2',
    retrival_database_batch_size: int = 512,
    normalize_embeddings: bool = True,
    pooling_method: str = 'mean'
) -> Chroma:
    """
    通过参数加载数据库

    Args:
        data_name_list: 数据集名称列表
        encoder_model_name: 编码器模型名称
        retrival_database_batch_size: 批处理大小
        normalize_embeddings: 是否归一化嵌入
        pooling_method: 池化方法

    Returns:
        Chroma 向量数据库实例
    """
    database_store_path = 'RetrievalBase'
    retrieval_name = '_'.join(data_name_list)
    if len(data_name_list) != 1:
        retrieval_name = 'mix_' + retrieval_name
    store_path = f"./{database_store_path}/{retrieval_name}/{encoder_model_name}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_local_embed_model(
        encoder_model_name,
        device,
        retrival_database_batch_size,
        normalize_embeddings,
        pooling_method
    )
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=store_path
    )
    return retrieval_database


def test_local_embedding():
    """测试本地嵌入模型功能"""
    print("=== 测试本地嵌入模型 ===")

    # 测试文本
    test_texts = [
        "This is a test sentence.",
        "Another example for embedding.",
        "Machine learning is fascinating."
    ]

    try:
        # 初始化模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        model = get_local_embed_model(
            encoder_model_name='all-MiniLM-L6-v2',
            device=device,
            retrival_database_batch_size=32
        )

        # 测试文档嵌入
        print("\n测试文档嵌入...")
        doc_embeddings = model.embed_documents(test_texts)
        print(f"文档嵌入维度: {len(doc_embeddings[0])}")
        print(f"嵌入数量: {len(doc_embeddings)}")

        # 测试查询嵌入
        print("\n测试查询嵌入...")
        query = "What is machine learning?"
        query_embedding = model.embed_query(query)
        print(f"查询嵌入维度: {len(query_embedding)}")

        # 计算相似度
        print("\n计算相似度...")
        from numpy import dot
        from numpy.linalg import norm

        for i, text in enumerate(test_texts):
            similarity = dot(doc_embeddings[i], query_embedding) / (norm(doc_embeddings[i]) * norm(query_embedding))
            print(f"'{text}' 与查询的相似度: {similarity:.4f}")

        print("\n✅ 测试成功！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='本地嵌入模型的检索数据库')
    parser.add_argument('--dataset_name', type=str, help='数据集名称')
    parser.add_argument('--encoder_model', type=str, default='all-MiniLM-L6-v2', help='编码器模型名称')
    parser.add_argument('--flag_mix', type=bool, default=False, help='是否混合数据集')
    parser.add_argument('--test', action='store_true', help='运行测试')
    args = parser.parse_args()

    if args.test:
        # 运行测试
        test_local_embedding()
    elif args.dataset_name:
        dataset_name = args.dataset_name
        encoder_model = args.encoder_model
        flag_mix = args.flag_mix

        # 预处理数据集
        if dataset_name.find('body') != -1 and not os.path.exists(os.path.join(DATA_ROOT, 'enron-mail-body')):
            pre_process_dataset('enron-mail', 'body')
        if dataset_name.find('strip') != -1 and not os.path.exists(os.path.join(DATA_ROOT, 'enron-mail-strip')):
            pre_process_dataset('enron-mail', 'strip')
        if dataset_name.find('train') != -1 and not os.path.exists(os.path.join(DATA_ROOT, dataset_name)):
            split_dataset(dataset_name[:-6])
        if dataset_name.find('test') != -1 and not os.path.exists(os.path.join(DATA_ROOT, dataset_name)):
            split_dataset(dataset_name[:-5])

        # 构建检索数据库
        if flag_mix is True:
            if dataset_name.find('enron-mail') != -1:
                construct_retrieval_database(
                    [dataset_name, 'wikitext-103'],
                    ['single_file', 'recursive_character'],
                    encoder_model
                )
            elif dataset_name.find('chatdoctor') != -1:
                construct_retrieval_database(
                    [dataset_name, 'wikitext-103'],
                    ['by_two_line_breaks', 'recursive_character'],
                    encoder_model
                )
        else:
            if dataset_name.find('enron-mail') != -1:
                construct_retrieval_database([dataset_name], ['single_file'], encoder_model)
            elif dataset_name.find('chatdoctor') != -1:
                construct_retrieval_database([dataset_name], ['by_two_line_breaks'], encoder_model)
    else:
        print("请提供 --dataset_name 参数或使用 --test 运行测试")