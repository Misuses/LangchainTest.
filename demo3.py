import os
import time
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.utilities import GoogleSearchAPIWrapper


# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "OpenAI API密钥"
# 设置Google API密钥
os.environ["GOOGLE_API_KEY"] = "Google API密钥"
os.environ["GOOGLE_CSE_ID"] = "Google CSE ID"


# 加载文档，使用 utf-8 编码
loader = TextLoader('data.txt', encoding='utf-8')
documents = loader.load()

# 文本分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 生成嵌入向量
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# 初始化语言模型和问答链
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

# 初始化 Google 搜索工具
search = GoogleSearchAPIWrapper()


def run_question_answer_system():
    while True:
        query = input("请输入你的问题（输入'退出'结束对话）：")
        if query == "退出":
            break
        docs = docsearch.similarity_search(query)
        if not docs:
            # 使用 Google 搜索工具获取信息
            search_results = search.run(query)
            # 将搜索结果作为新的文档
            new_docs = [{"page_content": search_results}]
            response = chain.run(input_documents=new_docs, question=query)
            print(f"结合 Google 搜索的回答：{response}")
        else:
            response = chain.run(input_documents=docs, question=query)
            print(f"根据文档的回答：{response}")
        time.sleep(1)


if __name__ == "__main__":
    run_question_answer_system()