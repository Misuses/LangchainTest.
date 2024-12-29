import os
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "OpenAI API密钥"

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