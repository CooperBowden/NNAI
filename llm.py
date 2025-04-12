from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.gemini import Gemini
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from transformers import AutoTokenizer
from huggingface_hub import login
import pandas as pd
import time
import os
from typing import Optional
import requests


class ChatEng:
    """create llm, memory chat store, qdrant client, and run initial engine creation and ingestion"""
    def __init__(self):
        # HuggingFace token to download model
        login(token="my_token")


        # Embed model to use for ingest, if model changed, delete files in index_storage and rerun ingest()
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = self.embed_model

        model = requests.get("http://localhost:8000/v1/models").json()
        model = model["data"][0]["id"]
        self.llm = OpenAILike(model=model,
                              api_base="http://localhost:8000/v1",
                              api_key="na",
                              max_tokens=512,
                              is_chat_model=True,
                              temperature=0.3,
                              )

        Settings.llm = self.llm
        Settings.chunk_size = 2560
        self.ingest()

    def ingest(self):
        """ document storing and vector storing, loads Excel sheet using pd, creates nodes list,
        turns nodes into vector store """
        print("SYSTEM: Loading index")
        file_path = os.path.join('data', 'city_code', 'city_code_of_ordinances.xlsx')
        start_time = time.time()
        try:
            storage_context = StorageContext.from_defaults(persist_dir="index_storage")
            self.index = load_index_from_storage(storage_context)
        except FileNotFoundError:
            print("SYSTEM: no index found, running ingest")
            nodes = []
            excel = pd.read_excel(io=file_path,
                                  header=0,
                                  sheet_name=None)
            sheets = excel.values()
            for sheet in sheets:
                for index, row in sheet.iterrows():
                    if (str(row.iloc[1])) != "nan":
                        node = TextNode(text=str(row.iloc[4]), id_=str(row.iloc[1]))
                    else:
                        # autogenerate node id if not specified
                        node = TextNode(text=str(row.iloc[4]))
                    node.metadata = {"url": row.iloc[0], "Title": row.iloc[2], "Subtitle": row.iloc[3]}
                    nodes.append(node)
            print(f"Loaded {len(nodes)} Nodes in {(time.time() - start_time):.2f} seconds")
            self.index = VectorStoreIndex(nodes, show_progress=True)
            self.index.storage_context.persist(persist_dir="index_storage")
        # Post processor list for chat engine to process returned nodes
        self.node_postprocessors = []
        self.node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.6))
        print(f"\nFinished ingest in {(time.time() - start_time):.2f} seconds")

    def create_engine(self, system_prompt=None):
        """creates chat engine with desired parameters/settings and memory per chat session"""
        print("Creating chat engine instance")
        chat_store = SimpleChatStore()
        memory = ChatMemoryBuffer.from_defaults(chat_store=chat_store, token_limit=4096)

        chat_eng = self.index.as_chat_engine(
            system_prompt=system_prompt,
            memory=memory,
            llm=self.llm,
            chat_mode="context",
            similarity_top_k=5,
            node_postprocessors=self.node_postprocessors,
            )
        return chat_eng

    def get_metadata(self, nodes):
        """ retrieve and return the metadata from nodes
        :param nodes: node data
        :return metadata: node metadata and score
        :rtype: dict
        """
        metadata = []
        for node in nodes:
            node.metadata["Score"] = node.get_score()
            metadata.append(node.metadata)
        return metadata

    def generate_questions(self, nodes, max_questions, query=None, question_prompt: Optional[str] = None,):
        """ generate suggested questions for each node
        :param nodes:
        :type nodes: TextNode List
        :param max_questions: max amount of questions to be generated
        :type max_questions: int
        :param query: user query
        :param question_prompt: prompt to generate questions
        :return: List of suggested questions
        """
        nodes = nodes[0:max_questions]
        question_prompt = (question_prompt or
                           ("You are an assistant. Your task is to setup 1 question "
                            "that the user might ask. Assume the user has no knowledge of anything in the context. "
                            "The questions should be diverse in nature across the "
                            "document. Restrict the questions to the context information provided."
                            "Do not reference the section or chapter. Make the question short"
                            "Make the questions similar to the user query, but not the same as the user query"
                            f"user query: {query}"))
        dataset_generator = RagDatasetGenerator.from_documents(
            documents=nodes,
            llm=self.llm,
            num_questions_per_chunk=1,
            question_gen_query=question_prompt
        )
        questions = dataset_generator.generate_questions_from_nodes()
        return questions

    def read_questions(self):
        """ read file of pre-generated questions """
        questions = []
        with open("data/questions.txt", "r") as fout:
            for question in fout.readlines():
                if "reserved" not in question:
                    questions.append(question.strip('\n'))
        return questions

    def write_questions(self):
        """ Generate questions for all nodes loaded into index and write to txt file. """
        question_prompt = ("You are an assistant. Your task is to setup 1 question "
                           "that the user might ask. Assume the user has no knowledge of the contents of the context. "
                           "Restrict the questions to the context information provided. "
                           "The question should be something the user might ask without having read the context. "
                           "Do not reference the section or chapter. Make questions short to medium length and "
                           "easy to understand"
                           "Avoid questions like: "
                           "What is the content about? "
                           "What can be found at the provided URL? "
                           )
        nodes = self.index.docstore.docs.values()
        nodes = list(nodes)
        # nodes = random.sample(nodes, 50)
        questions = self.generate_questions(nodes, question_prompt=question_prompt, max_questions=4)
        with open("data/questions.txt", "w+") as fin:
            for question in questions:
                question = question[1]
                for q in question:
                    fin.write(str(q.query) + "\n")

    def read_additional_resources(self):
        """ Read text file of resources to be returned as string and added to system prompt """
        with open("./data/resources.txt", "r") as fin:
            resources = fin.readlines()
            resources = " ".join(resources)
            return resources

    def ask_console(self, chat_eng):
        """ Used for testing in terminal,  NOT for production use """
        query = input("SYSTEM: Ask a question or type r to reset | type q to exit:\n")
        while query != "q":
            if query == "r":
                chat_eng.reset()
                print("SYSTEM: New conversation")
            else:
                start_time = time.time()
                print("SYSTEM: Generating Response")
                response_stream = chat_eng.stream_chat(query)
                response_stream.print_response_stream()
                source_nodes = response_stream.source_nodes
                metadata = self.get_metadata(source_nodes)
                print("\nReference: ", metadata)
                nodes = []
                for source_node in source_nodes:
                    nodes.append(source_node.node)
                questions = self.generate_questions(nodes, max_questions=1)
                print("Questions: ", questions)
                print("SYSTEM: end time:", round(time.time() - start_time))
            query = input("\nSYSTEM: Ask a question or type r to reset | type q to exit:\n")
