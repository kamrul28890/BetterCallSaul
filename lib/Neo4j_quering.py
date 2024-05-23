from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
import textwrap
import streamlit as st
# Warning control
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

VECTOR_INDEX_NAME = 'case_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'


NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT')
OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY')

print(OPENAI_API_KEY)

retrieval_query_window = """
MATCH window=
    (:Chunk)-[:NEXT*0..100]->(node)-[:NEXT*0..100]->(:Chunk)
WITH node, node.score AS score, window AS longestWindow 
ORDER BY length(window) DESC LIMIT 1
WITH nodes(longestWindow) AS chunkList, node, score
UNWIND chunkList AS chunkRow
OPTIONAL MATCH (chunkRow)-[:SIMILAR_CASE_TYPE]-(relatedChunk)
WITH chunkList, COLLECT(DISTINCT relatedChunk)[..10] AS relatedChunks, node, score
// Combine the original chunks with related chunks.
WITH chunkList + relatedChunks AS allChunks, node, score
UNWIND allChunks AS chunk
WITH COLLECT(DISTINCT chunk) AS uniqueChunks, node, score
UNWIND uniqueChunks AS chunkRow
WITH COLLECT(chunkRow.text) AS textList, score, node
RETURN apoc.text.join(textList, " \n ") AS text, score, node {.source} AS metadata
"""

vector_store_window = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database="neo4j",
    index_name=VECTOR_INDEX_NAME,
    text_node_property=VECTOR_SOURCE_PROPERTY,
    retrieval_query=retrieval_query_window,
)

# Create a retriever from the vector store
retriever_window = vector_store_window.as_retriever()

# Create a chatbot Question & Answer chain from the retriever
chain_window = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), 
    chain_type="stuff", 
    retriever=retriever_window
)

# question = "Give me a theft case and the sentence and what is the case you are referring to"

# answer = chain_window(
#     {"question": question},
#     return_only_outputs=True,
# )
# print(textwrap.fill(answer["answer"]))

st.header("The Law Chatbot")
st.subheader("@glassmates.legal")
question = st.text_input("Enter Your question", "Type here ...")

# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
	question_text = question.title()
	result = chain_window(
    {"question": question_text},
    return_only_outputs=True,
    )
	st.success(result['answer'])