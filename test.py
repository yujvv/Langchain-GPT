import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = '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'


# llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
# res = llm("怎么评价人工智能")
# print("reply____________" + res)


# loader = DirectoryLoader('/', glob='**/*.txt')
loader = DirectoryLoader('./', glob='data_t.txt')
# 将数据转成 document 对象
documents = loader.load()

# print("________________" + documents[0].content)


text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()
# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# 进行问答
result = qa({"query": "德国举行的科隆游戏展中，米哈游的绝零区的实机演示有多久？"})
print(result["result"])


# https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide
# 构建向量索引数据库

# chroma 是个本地的向量数据库，他提供的一个 persist_directory 来设置持久化目录进行持久化。读取时，只需要调取 from_document 方法加载即可。


