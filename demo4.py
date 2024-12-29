import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# 加载本地模型和 tokenizer
model_name = "G:\\Model\\Qwen2-0.5B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")


# 加载文档，使用 utf-8 编码
loader = TextLoader('data.txt', encoding='utf-8')
documents = loader.load()

# 文本分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 生成嵌入向量，显式指定模型名称
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Chroma.from_documents(texts, embeddings)

# 初始化问答链
chain = load_qa_chain(model, chain_type="stuff")


def run_question_answer_system():
    while True:
        query = input("请输入你的问题（输入'退出'结束对话）：")
        if query == "退出":
            break
        docs = docsearch.similarity_search(query)
        if not docs:
            print("很抱歉，没有找到与该问题相关的信息，无法回答。")
        else:
            response = chain.run(input_documents=docs, question=query)
            print(response)


if __name__ == "__main__":
    run_question_answer_system()