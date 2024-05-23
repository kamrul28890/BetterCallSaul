from Data_manipulation import return_chunk_data
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from Data_manipulation import return_transformed_data

load_dotenv()

first_case = return_chunk_data()

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

kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

merge_chunk_node_query = """
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.name = $chunkParam.name,
        mergedChunk.case_type = $chunkParam.case_type,
        mergedChunk.id = $chunkParam.id, 
        mergedChunk.decision_date = $chunkParam.decision_date, 
        mergedChunk.source = $chunkParam.source, 
        mergedChunk.court = $chunkParam.court,
        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId, 
        mergedChunk.text = $chunkParam.text
RETURN mergedChunk
"""

kg.query("""
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
""")

node_count = 0
for chunk in first_case:
    print(f"Creating `:Chunk` node for chunk ID {chunk['chunkId']}")
    kg.query(merge_chunk_node_query, 
            params={
                'chunkParam': chunk
            })
    node_count += 1
print(f"Created {node_count} nodes")


kg.query("""
         CREATE VECTOR INDEX `case_chunks` IF NOT EXISTS
          FOR (c:Chunk) ON (c.textEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'    
         }}
""")

kg.query("""
    MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
    WITH chunk, genai.vector.encode(
      chunk.text, 
      "OpenAI", 
      {
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint
      }) AS vector
    CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
    """, 
    params={"openAiApiKey": OPENAI_API_KEY, "openAiEndpoint": OPENAI_ENDPOINT} )


# def neo4j_vector_search(question):
#   """Search for similar nodes using the Neo4j vector index"""
#   vector_search_query = """
#     WITH genai.vector.encode(
#       $question, 
#       "OpenAI", 
#       {
#         token: $openAiApiKey,
#         endpoint: $openAiEndpoint
#       }) AS question_embedding
#     CALL db.index.vector.queryNodes($index_name, $top_k, question_embedding) yield node, score
#     RETURN score, node.name AS CaseName, node.text AS text, node.case_type as case_type, node.source AS source
#   """
#   similar = kg.query(vector_search_query, 
#                      params={
#                       'question': question, 
#                       'openAiApiKey':'your oipenai api key',
#                       'openAiEndpoint': OPENAI_ENDPOINT,
#                       'index_name':VECTOR_INDEX_NAME, 
#                       'top_k': 10})
#   return similar

transformed_data = return_transformed_data
Cases = [{key: value for key, value in case.items() if key != 'text'} for case in transformed_data]
merge_case_node_query = """
MERGE(c:case {Caseid: $chunkParam.id})
    ON CREATE SET 
        c.name = $chunkParam.name,
        c.decision_date = $chunkParam.decision_date, 
        c.source = $chunkParam.source, 
        c.court = $chunkParam.court
RETURN c
"""
node_count = 0
for case in Cases:
    print(f"Creating `:Case` node for case ID {case['id']}")
    kg.query(merge_case_node_query, 
            params={
                'chunkParam': case
            })
    node_count += 1
print(f"Created {node_count} nodes")

cypher = """
  MATCH (chunk_same_case:Chunk)
  WHERE chunk_same_case.id = $caseIdParam
  WITH chunk_same_case //{.id, .name, .chunkId, .chunkSeqId}
      ORDER BY chunk_same_case.chunkSeqId
  WITH collect(chunk_same_case) as chunk_list
      //Creating the relationship between nodes
      CALL apoc.nodes.link(
        chunk_list, 
        "NEXT", 
        {avoidDuplicates: true}
        )
  RETURN size(chunk_list)
"""
for case in Cases:
    kg.query(cypher, params={'caseIdParam': case['id']})

cypher = """
  MATCH (c:Chunk), (ca:case)
    WHERE c.id = ca.Caseid
  MERGE (c)-[newRelationship:PART_OF]->(ca)
  RETURN count(newRelationship)
"""

kg.query(cypher)

cypher = """
    MATCH (chunk1:Chunk), (chunk2:Chunk)
    WHERE chunk1.case_type = chunk2.case_type AND chunk1.chunkId <> chunk2.chunkId
    MERGE (chunk1)-[:SIMILAR_CASE_TYPE]->(chunk2)
"""

kg.query(cypher)


