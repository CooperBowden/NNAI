from llm import ChatEng
from llama_index.llms.gemini import Gemini
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator, BatchEvalRunner
import pandas as pd
import asyncio
import nest_asyncio
import csv
import time
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.azure_inference import AzureAICompletionsModel
import requests

import os

nest_asyncio.apply()

# For testing using Azure OpenAI API
os.environ["AZURE_INFERENCE_ENDPOINT"] = "your URL"
os.environ["AZURE_INFERENCE_CREDENTIAL"] = "your API key"
azure_openai = AzureOpenAI(
    model="gpt-4-0613",
    engine="NNAI",
    azure_endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    api_key=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    api_version="2024-05-01-preview",
    max_retries=10,
)


# create local llm instance
model = ChatEng()
llm = model.llm
index = model.index
model_name = requests.get("http://localhost:8000/v1/models").json()
model_name = model_name["data"][0]["id"]
local = OpenAILike(model=model_name,
                          api_base="http://localhost:8000/v1",
                          api_key="na",
                          is_chat_model=True,
                          )

# set what LLM is used as the evaluator (local or azure_openai)
evaluator_llm = azure_openai

async def evaluate_query_engine(query_engine, qas, feval, reval, ceval):
    runner = BatchEvalRunner(
        {"faithfulness": feval, "relevancy": reval, "correctness": ceval},
        workers=3, show_progress=True
    )
    eval_results = await runner.aevaluate_queries(
        query_engine,
        queries=qas[0],
        reference=qas[1],
    )


    return eval_results

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    correct_score = 0
    with open(f"data/evaluation_{key}.csv", "w",  encoding="utf-8", newline='', errors="ignore") as fin:
        writer = csv.writer(fin, delimiter=";")
        writer.writerow(["model:", "deepseek-r1", "embeddings model:", model.embed_model, "node postprocessors:", model.node_postprocessors, "prompt:", system_prompt, "similarity_top_k:", top_k, "evaluated by:", evaluator_llm.model])
        writer.writerow(["query", "response", "passing", "score", "feedback", "invalid_result", "invalid_reason", "context"])
        for result in results:
            writer.writerow(
                [result.query, result.response, result.passing, result.score, result.feedback, result.invalid_result,
                 result.invalid_reason, result.contexts])
            if (key == "correctness"):
                correct_score += result.score
            if result.passing:
                correct += 1
    score = correct / len(results)
    print(f"{key} pass percent: {score}")
    if (key == "correctness"):
        print("correctness avg Score: " + str(correct_score/len(results)) + "/5")
    return score


if __name__ == "__main__":
    start_time = time.time()
    faithfulness = FaithfulnessEvaluator(llm=evaluator_llm)
    relevancy = RelevancyEvaluator(llm=evaluator_llm)
    correctness = CorrectnessEvaluator(llm=evaluator_llm)
    system_prompt = ("I am a citizen of Newport News, Virginia asking about the city code. "
                     "You are a chatbot for the city of Newport News that answers the user's question. "
                     "Using only the provided context, answer the user's query and provide all relevant information. "
                     "If the answer cannot be found within the given context, "
                     "state that you do not have enough information. "
                     )

    with open(f"data/qa_pairs.csv", "r+",  encoding="utf-8", newline='', errors="ignore") as fin:
        reader = csv.reader(fin, delimiter=";")
        questions = []
        r_answers = []
        for line in reader:
            questions.append(str(line[0]))
            r_answers.append(str(line[1]))
        qas = [questions, r_answers]
    top_k = 5
    query_engine = index.as_query_engine(llm=llm,
                                         similarity_top_k=top_k,
                                         node_postprocessors=model.node_postprocessors,
                                         system_prompt=system_prompt,
                                        )

    result = asyncio.run(evaluate_query_engine(query_engine, qas, faithfulness, relevancy, correctness))
    score = get_eval_results("faithfulness", result)
    score = get_eval_results("relevancy", result)
    score = get_eval_results("correctness", result)
    print(f"finished eval in {(time.time()-start_time):.2f}s")
