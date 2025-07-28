import json
import re

import requests
import time
import pandas as pd
import os
import argparse
from jsonlines import jsonlines
from openai import OpenAI

from elasticsearch import Elasticsearch, NotFoundError


Parser = argparse.ArgumentParser()
Parser.add_argument("--domain", default="logistics", help="which domain to evaluate", choices=["blocksworld", "mystery_blocksworld", "barman", "logistics"])
Parser.add_argument("--model", default="deepseek/deepseek-r1-distill-llama-70b", help="which model to use", choices=["deepseek/deepseek-r1-distill-llama-70b","deepseek/deepseek-r1-distill-qwen-14b","qwen/qwen3-8b","qwen/qwq-32b","meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct","deepseek-r1-distill-qwen-14b","deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "qwen2.5-coder-7b-instruct", "qwq-32b", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-preview", "google/gemma-2-9b-it", "google/gemma-2-27b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "o3-mini", "meta-llama/Llama-3.1-405B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
Parser.add_argument("--data", default="Heavily_Templated_Logistics-100", help="which data to use", choices=["Moderately_Templated_BlocksWorld-100", "Heavily_Templated_BlocksWorld-111", "Moderately_Templated_BlocksWorld-111", "Natural_BlocksWorld-111", "Heavily_Templated_Mystery_BlocksWorld-100", "Heavily_Templated_Barman-100", "Heavily_Templated_Logistics-100", "Moderately_Templated_Logistics-100", "Natural_Logistics-100", "Heavily_Templated_BlocksWorld-100"])
Parser.add_argument("--index_start", default="33", help="index to start generating result from (inclusive)")
Parser.add_argument("--index_end", default="51", help="index to end generating result from (exclusive)")
Parser.add_argument("--solver", help="which solver to use", default="dual-bfws-ffparser")





# 配置 Elasticsearch 连接
elastic_user = "elastic"
elastic_password = ""
elastic_endpoint = "localhost"

url = f"https://{elastic_user}:{elastic_password}@{elastic_endpoint}:9200"
es = Elasticsearch(url, verify_certs=False)


# 1. 创建索引并配置 BM25 参数
def create_index(name):
    index_name = name

    # 检查索引是否已存在
    if not es.indices.exists(index=index_name):
        # 如果索引不存在，创建它
        es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "index": {
                        "similarity": {
                            "default": {
                                "type": "BM25",  # 指定相似度类型为 BM25
                                "k1": 1.5,  # k1 参数
                                "b": 0.75  # b 参数
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",  # 使用文本类型
                        },
                        "content": {
                            "type": "text"
                        }
                    }
                }
            }
        )
        print("Index created successfully!")
    else:
        print("Index already exists!")


# 2. 向索引添加文档
def add_documents(name):
    index_name = name
    documents = []

    # 读取文档数据
    with open(f"{name}.jsonl", 'r', encoding='utf-8') as f:
        for data in jsonlines.Reader(f):
            documents.append(data)

    # 检查索引是否存在，如果不存在则创建索引
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)

    # 遍历文档并检查是否存在，如果不存在则添加
    for i, doc in enumerate(documents):
        doc_id = i + 1
        if not es.exists(index=index_name, id=doc_id):
            es.index(index=index_name, id=doc_id, document=doc)
            print(f"Document {doc_id} added.")
        else:
            print(f"Document {doc_id} already exists.")

    print("Documents processed successfully!")


# 3. 手动刷新索引
def refresh_index():
    es.indices.refresh(index="my_index")
    print("Index refreshed successfully!")


# 4. 检查索引中的文档
def check_documents(name):
    response = es.search(
        index=name,
        body={
            "query": {
                "match_all": {}  # 查询所有文档
            },
            "size": 10  # 获取最多10个文档
        }
    )
    print("Documents in the index:")
    for hit in response['hits']['hits']:
        print(hit['_source'])


# 5. 执行查询并使用 BM25 进行搜索
def search_documents(query, k, name):
    # 执行检索查询，查询 content 和 title 字段
    response = es.search(
        index=name,
        body={
            "query": {
                "multi_match": {  # 使用 multi_match 查询多个字段
                    "query": query,
                    "fields": ["example"],  # 查询 title 和 content 字段
                }
            },
            "size": k,  # 限制返回文档数量为 k
            "_source": ["type_name", "document", "example"],  # 返回 title 和 content 字段
            "sort": [{"_score": {"order": "desc"}}]  # 按照得分降序排序
        }
    )

    # 获取 title 和 content 字段并组合成列表
    combined_list = []
    hits = response['hits']['hits']
    if hits:
        for hit in hits:
            type_name = hit['_source'].get('type_name', '')  # 获取 title，如果不存在则返回空字符串
            documentation = hit['_source'].get('document', '')  # 获取 content，如果不存在则返回空字符串
            example = hit['_source'].get('example', '')  # 获取 content，如果不存在则返回空字符串
            combined_text = f"type_name: {type_name}\ndocumentation: {documentation}\n{type_name} example:\n{example}"  # 将 title 和 content 组合成一个字符串
            combined_list.append(combined_text)

    return combined_list


create_index("domain_data")  # 创建索引
create_index("problem_data")  # 创建索引
add_documents("domain_data")  # 添加文档
add_documents("problem_data")  # 添加文档


prompt_version = "pddl_instruction"

args = Parser.parse_args()
MODEL = args.model

if "gpt" in MODEL:
    # OPENAI_API_KEY = open(f'../../../_private/key.txt').read()
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    client = OpenAI(api_key=OPENAI_API_KEY)
elif MODEL in ["deepseek-reasoner"]:
    DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
elif MODEL in ["qwen2.5-coder-7b-instruct", "qwq-32b", "qwen3-8b", "deepseek-r1-distill-qwen-7b",
               "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-llama-70b", "deepseek-r1-distill-qwen-14b",
               "deepseek-r1-distill-llama-70b"]:
    QWEN_API_KEY = "YOUR_QWEN_API_KEY"
    client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
elif MODEL in ["deepseek/deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-qwen-14b", "qwen/qwen3-8b",
               "qwen/qwq-32b", "meta-llama/llama-4-maverick-17b-128e-instruct",
               "meta-llama/llama-4-scout-17b-16e-instruct"]:
    Router_API_KEY = "YOUR_Router_API_KEY"
    client = OpenAI(api_key=Router_API_KEY, base_url="https://openrouter.ai/api/v1")

def query_reformer_gpt(domain, data, problem, model, result, count, force_json=False):

    if "meta" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        _, model_name = model.split("/")

    output_format = "json_object" if force_json else "text"

    if model in ["meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct"]:
        if count == 0:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf.pddl').read()
        else:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df_{count - 1}.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf_{count - 1}.pddl').read()
    else:
        if count == 0:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf.pddl').read()
        else:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()

    # if "line" in result:
    #     pattern = r"\d+"  # 匹配一个或多个数字
    #     result = re.search(pattern, result)
    #     line = int(result.group())
    #
    #     query = previous_domain_file.split("\n")[line - 1:line + 5]
    #     query = "\n".join(query)
    #     return query
    if "domain" not in result:
        prompt = f'''
I provided a set of PDDL files and their errors. Please extract the full error section where the error code is located. Only give me the error code, do not add any other word.

Here are some examples:
Wrong_domain_file:
(define (domain block-stacking)
  (:requirements :strips :typing)

  (:types block - object)

  (:predicates
    (clear ?b - block)
    (block-on-table ?b - block)
    (arm-empty)
    (holding ?b - block)
  )

  (:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )

  (:action putdown
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (clear ?b) (block-on-table ?b) arm-empty (not (holding ?b)))
  )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (clear ?b2) (holding ?b1))
    :effect (and arm-empty (clear ?b1) (block-on-table ?b1) (not (clear ?b2)) (not (holding ?b1)))
  )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (block-on-table ?b1) (clear ?b1) arm-empty)
    :effect (and (holding ?b1) (clear ?b2) (not (block-on-table ?b1)) (not (clear ?b1)) (not arm-empty))
  ))

Wrong_problem_file:
(define (problem block-stacking-problem)
  (:domain block-stacking)

  (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)

  (:init
    (clear b3) (clear b4) (clear b8) arm-empty
    (block-on-table b1) (block-on-table b2) (block-on-table b3)
    (block-on-table b4) (block-on-table b5) (block-on-table b6)
    (block-on-table b7) (block-on-table b8)
    (on-top-of b4 b6) (on-top-of b5 b1) (on-top-of b6 b2)
    (on-top-of b7 b5) (on-top-of b8 b7)
  )

  (:goal
    (and (clear b6) (block-on-table b6) (clear b5) (block-on-table b5)
         (clear b1) (block-on-table b1) (clear b2) (block-on-table b2)
         (clear b3) (block-on-table b3) (clear b4) (block-on-table b4)
         (clear b5) (block-on-table b5) (clear b7) (block-on-table b7)
         (clear b8) (block-on-table b8))
  ))

error:
domain: syntax error in line 15, 'ARM-EMPTY':

error code: 
(:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )


Wrong_domain_file:
(define (domain block-stacking)
(:requirements :strips :typing)

(:types block - object
     table - object
     hand - object)

(:predicates
(clear ?x - block)
(on-table ?x - block)
(holding ?x - block)
(block-on-block ?x ?y - block))

(:action pickup
:parameters (?x - block)
:precondition (and (clear ?x) (on-table ?x) (hand-empty))
:effect (and (not (clear ?x)) (not (on-table ?x)) (not (hand-empty)) (holding ?x)))
)

(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

(:action stack
:parameters (?x ?y - block)
:precondition (and (clear ?x) (holding ?y))
:effect (and (not (clear ?x)) (not (holding ?y)) (hand-empty) (clear ?y) (block-on-block ?y ?x)))
)

(:action unstack
:parameters (?x ?y - block)
:precondition (and (block-on-block ?x ?y) (clear ?x) (hand-empty))
:effect (and (holding ?x) (not (clear ?y)) (not (block-on-block ?x ?y)) (clear ?x))
))
)

Wrong_problem_file:
(define (problem block-stacking-problem)
(:domain block-stacking)

(:objects
b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 t1 - block
table - table
hand - hand)

(:init
(clear b1) (clear b2) (clear b3) (clear b4) (clear b6) (clear b7) (clear b8) (clear b9) (clear b10) (clear b11) (clear b12) (clear b13) (clear b14) (clear b15)
(on-table b1) (on-table b2) (on-table b3) (on-table b4) (on-table b5) (on-table b6) (on-table b7) (on-table b8) (on-table b9) (on-table b10) (on-table b11) (on-table b12) (on-table b13) (on-table b14) (on-table b15)
(hand-empty)
(block-on-block b15 t1) (block-on-block b13 t1) (block-on-block b9 t1) (block-on-block b5 t1)
(block-on-block b3 t1) (block-on-block b6 t1) (block-on-block b7 t1) (block-on-block b8 t1) (block-on-block b10 t1) (block-on-block b11 t1) (block-on-block b12 t1) (block-on-block b14 t1)
)

(:goal
(and
(block-on-block b11 b15)
(block-on-block b13 b5)
(on-table b1)
(on-table b2)
(on-table b3)
(on-table b4)
(on-table b5)
(on-table b6)
(on-table b7)
(on-table b8)
(on-table b9)
(on-table b10)
(on-table b11)
(on-table b12)
(on-table b13)
(on-table b14)
(on-table b15)
)
))
error:
domain: syntax error in line 20, '(':

error code: 
(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

error:
domain: syntax error in line 20, '(':

error code: 
(:action putdown
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

Only generate the error code below
Wrong_domain_file:
{previous_domain_file}

Wrong_problem_file:
{previous_problem_file}

error:\n{result}

error code: 
'''
    else:
        prompt = f'''
I provided a set of PDDL files and their errors. Please extract the full error section. Only give me the error code, do not add any other word.

Here are some examples:
Wrong_domain_file:
(define (domain block-stacking)
  (:requirements :strips :typing)

  (:types block - object)

  (:predicates
    (clear ?b - block)
    (block-on-table ?b - block)
    (arm-empty)
    (holding ?b - block)
  )

  (:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )

  (:action putdown
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (clear ?b) (block-on-table ?b) arm-empty (not (holding ?b)))
  )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (clear ?b2) (holding ?b1))
    :effect (and arm-empty (clear ?b1) (block-on-table ?b1) (not (clear ?b2)) (not (holding ?b1)))
  )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (block-on-table ?b1) (clear ?b1) arm-empty)
    :effect (and (holding ?b1) (clear ?b2) (not (block-on-table ?b1)) (not (clear ?b1)) (not arm-empty))
  ))

error:
domain: syntax error in line 15, 'ARM-EMPTY':

error code: 
(:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )


Wrong_domain_file:
(define (domain block-stacking)
(:requirements :strips :typing)

(:types block - object
     table - object
     hand - object)

(:predicates
(clear ?x - block)
(on-table ?x - block)
(holding ?x - block)
(block-on-block ?x ?y - block))

(:action pickup
:parameters (?x - block)
:precondition (and (clear ?x) (on-table ?x) (hand-empty))
:effect (and (not (clear ?x)) (not (on-table ?x)) (not (hand-empty)) (holding ?x)))
)

(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

(:action stack
:parameters (?x ?y - block)
:precondition (and (clear ?x) (holding ?y))
:effect (and (not (clear ?x)) (not (holding ?y)) (hand-empty) (clear ?y) (block-on-block ?y ?x)))
)

(:action unstack
:parameters (?x ?y - block)
:precondition (and (block-on-block ?x ?y) (clear ?x) (hand-empty))
:effect (and (holding ?x) (not (clear ?y)) (not (block-on-block ?x ?y)) (clear ?x))
))
)

error:
domain: syntax error in line 20, '(':

error code: 
(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

Only generate the error code below
Wrong_domain_file:
{previous_domain_file}

error:\n{result}

error code: 
'''

    # message = prompt + "Return a JSON object in the following format:\n{\n  \"error code\": ...\n}"

    # print(message)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        # response_format={"type": output_format}
    )

    return_string = completion.choices[0].message.content
    print(return_string)
    return return_string

def query_reformer_qwen(domain, data, problem, model, result, count, force_json=False):
    if model in ["qwen/qwq-32b","qwen/qwen3-8b"]:
        _, model_name = model.split("/")
    else:
        model_name = model
    output_format = "json_object" if force_json else "text"

    if count == 0:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf.pddl').read()
    else:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()


    # if "line" in result:
    #     pattern = r"\d+"  # 匹配一个或多个数字
    #     result = re.search(pattern, result)
    #     line = int(result.group())
    #
    #     query = previous_domain_file.split("\n")[line - 1:line + 5]
    #     query = "\n".join(query)
    #     return query
    if "domain" not in result:
        prompt = f'''
I provided a set of PDDL files and their errors. Please extract the full error section where the error code is located. Only give me the error code, do not add any other word.

Here are some examples:
Wrong_domain_file:
(define (domain block-stacking)
  (:requirements :strips :typing)

  (:types block - object)

  (:predicates
    (clear ?b - block)
    (block-on-table ?b - block)
    (arm-empty)
    (holding ?b - block)
  )

  (:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )

  (:action putdown
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (clear ?b) (block-on-table ?b) arm-empty (not (holding ?b)))
  )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (clear ?b2) (holding ?b1))
    :effect (and arm-empty (clear ?b1) (block-on-table ?b1) (not (clear ?b2)) (not (holding ?b1)))
  )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (block-on-table ?b1) (clear ?b1) arm-empty)
    :effect (and (holding ?b1) (clear ?b2) (not (block-on-table ?b1)) (not (clear ?b1)) (not arm-empty))
  ))

Wrong_problem_file:
(define (problem block-stacking-problem)
  (:domain block-stacking)

  (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)

  (:init
    (clear b3) (clear b4) (clear b8) arm-empty
    (block-on-table b1) (block-on-table b2) (block-on-table b3)
    (block-on-table b4) (block-on-table b5) (block-on-table b6)
    (block-on-table b7) (block-on-table b8)
    (on-top-of b4 b6) (on-top-of b5 b1) (on-top-of b6 b2)
    (on-top-of b7 b5) (on-top-of b8 b7)
  )

  (:goal
    (and (clear b6) (block-on-table b6) (clear b5) (block-on-table b5)
         (clear b1) (block-on-table b1) (clear b2) (block-on-table b2)
         (clear b3) (block-on-table b3) (clear b4) (block-on-table b4)
         (clear b5) (block-on-table b5) (clear b7) (block-on-table b7)
         (clear b8) (block-on-table b8))
  ))

error:
domain: syntax error in line 15, 'ARM-EMPTY':

error code: 
(:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )


Wrong_domain_file:
(define (domain block-stacking)
(:requirements :strips :typing)

(:types block - object
     table - object
     hand - object)

(:predicates
(clear ?x - block)
(on-table ?x - block)
(holding ?x - block)
(block-on-block ?x ?y - block))

(:action pickup
:parameters (?x - block)
:precondition (and (clear ?x) (on-table ?x) (hand-empty))
:effect (and (not (clear ?x)) (not (on-table ?x)) (not (hand-empty)) (holding ?x)))
)

(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

(:action stack
:parameters (?x ?y - block)
:precondition (and (clear ?x) (holding ?y))
:effect (and (not (clear ?x)) (not (holding ?y)) (hand-empty) (clear ?y) (block-on-block ?y ?x)))
)

(:action unstack
:parameters (?x ?y - block)
:precondition (and (block-on-block ?x ?y) (clear ?x) (hand-empty))
:effect (and (holding ?x) (not (clear ?y)) (not (block-on-block ?x ?y)) (clear ?x))
))
)

Wrong_problem_file:
(define (problem block-stacking-problem)
(:domain block-stacking)

(:objects
b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 t1 - block
table - table
hand - hand)

(:init
(clear b1) (clear b2) (clear b3) (clear b4) (clear b6) (clear b7) (clear b8) (clear b9) (clear b10) (clear b11) (clear b12) (clear b13) (clear b14) (clear b15)
(on-table b1) (on-table b2) (on-table b3) (on-table b4) (on-table b5) (on-table b6) (on-table b7) (on-table b8) (on-table b9) (on-table b10) (on-table b11) (on-table b12) (on-table b13) (on-table b14) (on-table b15)
(hand-empty)
(block-on-block b15 t1) (block-on-block b13 t1) (block-on-block b9 t1) (block-on-block b5 t1)
(block-on-block b3 t1) (block-on-block b6 t1) (block-on-block b7 t1) (block-on-block b8 t1) (block-on-block b10 t1) (block-on-block b11 t1) (block-on-block b12 t1) (block-on-block b14 t1)
)

(:goal
(and
(block-on-block b11 b15)
(block-on-block b13 b5)
(on-table b1)
(on-table b2)
(on-table b3)
(on-table b4)
(on-table b5)
(on-table b6)
(on-table b7)
(on-table b8)
(on-table b9)
(on-table b10)
(on-table b11)
(on-table b12)
(on-table b13)
(on-table b14)
(on-table b15)
)
))
error:
domain: syntax error in line 20, '(':

error code: 
(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

Only generate the error code below
Wrong_domain_file:
{previous_domain_file}

Wrong_problem_file:
{previous_problem_file}

error:\n{result}

error code: 
'''
    else:
        prompt = f'''
I provided a set of PDDL files and their errors. Please extract the full error section. Only give me the error code, do not add any other word.
Here are some examples:
Wrong_domain_file:
(define (domain block-stacking)
  (:requirements :strips :typing)

  (:types block - object)

  (:predicates
    (clear ?b - block)
    (block-on-table ?b - block)
    (arm-empty)
    (holding ?b - block)
  )

  (:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )

  (:action putdown
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and (clear ?b) (block-on-table ?b) arm-empty (not (holding ?b)))
  )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (clear ?b2) (holding ?b1))
    :effect (and arm-empty (clear ?b1) (block-on-table ?b1) (not (clear ?b2)) (not (holding ?b1)))
  )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (block-on-table ?b1) (clear ?b1) arm-empty)
    :effect (and (holding ?b1) (clear ?b2) (not (block-on-table ?b1)) (not (clear ?b1)) (not arm-empty))
  ))

error:
domain: syntax error in line 15, 'ARM-EMPTY':

error code: 
(:action pickup
    :parameters (?b - block)
    :precondition (and (clear ?b) (block-on-table ?b) arm-empty)
    :effect (and (holding ?b) (not (clear ?b)) (not (block-on-table ?b)) (not arm-empty))
  )
 

Wrong_domain_file:
(define (domain block-stacking)
(:requirements :strips :typing)

(:types block - object
     table - object
     hand - object)

(:predicates
(clear ?x - block)
(on-table ?x - block)
(holding ?x - block)
(block-on-block ?x ?y - block))

(:action pickup
:parameters (?x - block)
:precondition (and (clear ?x) (on-table ?x) (hand-empty))
:effect (and (not (clear ?x)) (not (on-table ?x)) (not (hand-empty)) (holding ?x)))
)

(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

(:action stack
:parameters (?x ?y - block)
:precondition (and (clear ?x) (holding ?y))
:effect (and (not (clear ?x)) (not (holding ?y)) (hand-empty) (clear ?y) (block-on-block ?y ?x)))
)

(:action unstack
:parameters (?x ?y - block)
:precondition (and (block-on-block ?x ?y) (clear ?x) (hand-empty))
:effect (and (holding ?x) (not (clear ?y)) (not (block-on-block ?x ?y)) (clear ?x))
))
)

error:
domain: syntax error in line 20, '(':

error code: 
(:action putdown
:parameters (?x - block)
:precondition (holding ?x)
:effect (and (clear ?x) (on-table ?x) (hand-empty) (not (holding ?x))))
)

Only generate the error code below
Wrong_domain_file:
{previous_domain_file}

error:\n{result}

error code: 
'''

    # message = prompt + "Return a JSON object in the following format:\n{\n  \"error code\": ...\n}"

    # print(message)

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复
    print(prompt)
    # 创建聊天完成请求

    # completion = client.chat.completions.create(
    #     extra_body={},
    #     model=model,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ]
    # )
    # print(completion.choices[0].message.content)

    completion = client.chat.completions.create(
        model=model,  # 此处以 qwq-32b 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": prompt}
        ],
        # QwQ 模型仅支持流式输出方式调用
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering is False:
                    is_answering = True
                answer_content += delta.content

    return_string = answer_content
    # return_string = completion.choices[0].message.content


    # if model in ['o1-preview', 'deepseek-reasoner', 'qwen2.5-coder-7b-instruct']:
    #     start_index = return_string.find('{')
    #     end_index = return_string.find('}')
    #     json_string = return_string[start_index:end_index + 1]
    #     return_dict = json.loads(json_string, strict=False)
    # else:
    #     return_dict = json.loads(return_string, strict=False)
    #
    # error_code = return_dict["error code"]
    return return_string


def run_error_refine_gpt(domain, data, problem, model, result, count, query=None, doc=None, force_json=False):

    if "meta" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        _, model_name = model.split("/")

    force_json = True
    output_format = "json_object" if force_json else "text"

    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt', encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt', encoding="utf-8").read()

    if model in ["meta-llama/llama-4-maverick-17b-128e-instruct","meta-llama/llama-4-scout-17b-16e-instruct"]:
        if count == 0:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf.pddl').read()
        else:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df_{count - 1}.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf_{count - 1}.pddl').read()
    else:
        if count == 0:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf.pddl').read()
        else:
            previous_domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
            previous_problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()


    if "tree" in result:
        prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes. Do not add any other word.\n"

        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n}"

    elif "problem:" in result:
        doc = "\n\n\n".join(doc)

        prompt = f'''Knowledge:{doc}

Wrong_domain_file:
{previous_domain_file}
wrong_problem_file:
{previous_problem_file}
Wrong PDDL:
{query}

error: {result}

Instruction: I provided a wrong PDDL files and the documentation for the errors, you need according to the documentation, give me the corrected domain_file. You must make changes, and give me a logical reason for why you change like that. Do not add any other word.
'''
        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    elif "domain:" in result:
        doc = "\n\n\n".join(doc)

        # prompt = f"PDDL Syntax Examples:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nerror feedback:{result}\n\nerror location: {query}\n\nInstruction: I provided a wrong PDDL files, its error location and some PDDL Syntax Examples, you need according to the error feedback and the documentation, give me the corrected domain_file. You must make changes.\n"
        prompt = f'''Knowledge:{doc}
    
Wrong_domain_file:
{previous_domain_file}
Wrong PDDL:
{query}

error: {result}

Instruction: I provided a wrong PDDL files and the documentation for the errors, you need according to the documentation, give me the corrected domain_file. You must make changes, and give me a logical reason for why you change like that. Do not add any other word.
'''
        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    else:
        prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes.\n"

        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n}"

    #
    # prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"domain_description:\n{domain_description}\n\nproblem_description:\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\ndoc and example:\n{doc}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback and doc and example, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nerror location:{query}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"PDDL Syntax Examples:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes.\n"
    # prompt = f"{doc}\n\n\nYou are a PDDL expert. Here is a game we are playing.\n{domain_description}\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nerror location:{query}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes.\n"

    # message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n  \"reason\":...\n}"
    # message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    print(message)

    completion = client.chat.completions.create(
        model=model,

        messages=[
            {"role": "system", "content": message},
            {"role": "user", "content": message}
        ],

        response_format={"type": output_format}

    )

    return_string = completion.choices[0].message.content

    print(return_string)

    if model in ["deepseek/deepseek-r1-distill-llama-70b","deepseek/deepseek-r1-distill-qwen-14b","qwen/qwen3-8b","qwen/qwq-32b","meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct",
                 'o1-preview', 'deepseek-reasoner', "deepseek-r1-distill-llama-70b", "deepseek-r1-distill-qwen-7b",
                 "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-14b", "qwen2.5-coder-7b-instruct",
                 "qwen3-8b", "deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct",
                 "deepseek-r1-distill-llama-70b"]:

        start_index = return_string.find('{')
        end_index = return_string.find('}')
        json_string = return_string[start_index:end_index + 1]
        return_dict = json.loads(json_string, strict=False)
    else:
        return_dict = json.loads(return_string, strict=False)

    if "domain" in result:
        domain_file = return_dict["corrected domain file"]
        problem_file = previous_problem_file

        if "predicate" in result:
            j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file,
                 "problem_file": problem_file}
            with open("predicate_error_qwq.jsonl", "a", encoding="utf-8") as f:
                json.dump(j, f, indent=4)
                f.write("\n")
        else:
            j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file,
                 "problem_file": problem_file}
            with open("domain_syntax_error_qwq.jsonl", "a", encoding="utf-8") as f:
                json.dump(j, f, indent=4)
                f.write("\n")
    else:
        domain_file = previous_domain_file
        problem_file = return_dict["corrected problem file"]
        j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file,
             "problem_file": problem_file}
        with open("problem_error_qwq.jsonl", "a", encoding="utf-8") as f:
            json.dump(j, f, indent=4)
            f.write("\n")


    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count}.pddl'

    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count}.pddl'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file


def run_error_refine_qwen(domain, data, problem, model, result, count, query=None, doc=None, force_json=False):
    force_json = True
    if model in ["qwen/qwq-32b","qwen/qwen3-8b"]:
        _, model_name = model.split("/")
    else:
        model_name = model
    output_format = "json_object" if force_json else "text"

    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt', encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt', encoding="utf-8").read()

    if count == 0:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf.pddl').read()
    else:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()

    if "tree" in result:
        prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes. Do not add any other word.\n"

        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    elif "problem:" in result:
        doc = "\n\n\n".join(doc)

        prompt = f'''Knowledge:{doc}

Wrong_domain_file:
{previous_domain_file}
wrong_problem_file:
{previous_problem_file}
Wrong PDDL:
{query}

error: {result}

Instruction: I provided a wrong PDDL files and the documentation for the errors, you need according to the documentation, give me the corrected domain_file. You must make changes, and give me a logical reason for why you change like that. Do not add any other word.
'''
        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    elif "domain:" in result:
        doc = "\n\n\n".join(doc)

        # prompt = f"PDDL Syntax Examples:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nerror feedback:{result}\n\nerror location: {query}\n\nInstruction: I provided a wrong PDDL files, its error location and some PDDL Syntax Examples, you need according to the error feedback and the documentation, give me the corrected domain_file. You must make changes.\n"
        prompt = f'''Knowledge:{doc}

Wrong_domain_file:
{previous_domain_file}
Wrong PDDL:
{query}

error: {result}

Instruction: I provided a wrong PDDL files and the documentation for the errors, you need according to the documentation, give me the corrected domain_file. You must make changes, and give me a logical reason for why you change like that. Do not add any other word.
'''
        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n}"

    else:
        prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes.\n"

        message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    #
    # prompt = f"Wrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"domain_description:\n{domain_description}\n\nproblem_description:\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\ndoc and example:\n{doc}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback and doc and example, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"knowledge:{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nerror location:{query}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"PDDL Syntax Examples:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files and some PDDL Syntax Examples, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes.\n"
    # prompt = f"{doc}\n\n\nYou are a PDDL expert. Here is a game we are playing.\n{domain_description}\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nerror location:{query}\n\nInstruction: I provided a wrong set of PDDL files and the documentation for the errors, you need according to the error feedback and the documentation, give me the corrected domain_file and problem_file. You must make changes.\n"

    # message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n  \"reason\":...\n}"
    # message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n}"

    print(message)
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复
    # 创建聊天完成请求

    # completion = client.chat.completions.create(
    #     extra_body={},
    #     model=model,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     response_format={"type": output_format}
    # )
    # print(completion.choices[0].message.content)

    completion = client.chat.completions.create(
        model=model,  # 此处以 qwq-32b 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": message}
        ],
        # QwQ 模型仅支持流式输出方式调用
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
            else:
                if delta.content != "" and is_answering is False:
                    is_answering = True
                answer_content += delta.content

    return_string = answer_content
    # return_string = completion.choices[0].message.content

    print(return_string)

    if model in ["qwen/qwen3-8b","qwen/qwq-32b", 'o1-preview', 'deepseek-reasoner', "qwen2.5-coder-7b-instruct", "qwq-32b", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "llama-4-scout-17b-16e-instruct"]:
        start_index = return_string.find('{')
        end_index = return_string.find('}')
        json_string = return_string[start_index:end_index + 1]
        return_dict = json.loads(json_string, strict=False)
    else:
        return_dict = json.loads(return_string, strict=False)


    if "domain" in result :
        domain_file = return_dict["corrected domain file"]
        problem_file = previous_problem_file

        if "predicate" in result:
            j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file, "problem_file": problem_file}
            with open("predicate_error_qwq.jsonl", "a", encoding="utf-8") as f:
                json.dump(j, f, indent=4)
                f.write("\n")
        else:
            j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file, "problem_file": problem_file}
            with open("domain_syntax_error_qwq.jsonl", "a", encoding="utf-8") as f:
                json.dump(j, f, indent=4)
                f.write("\n")
    else:
        domain_file = previous_domain_file
        problem_file = return_dict["corrected problem file"]
        j = {"problem": problem, "message": message, "error": result, "domain_file": domain_file,
             "problem_file": problem_file}
        with open("problem_error_qwq.jsonl", "a", encoding="utf-8") as f:
            json.dump(j, f, indent=4)
            f.write("\n")
    # domain_file = return_dict["corrected domain file"]
    # problem_file = return_dict["corrected problem file"]

    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count}.pddl'

    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count}.pddl'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file


def run_solver(domain, data, problem, model, solver, count):
    if "meta" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        _, model_name = model.split("/")

    # domain_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_df.pddl').read()
    # problem_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_pf.pddl').read()

    if model in ["meta-llama/llama-4-maverick-17b-128e-instruct","meta-llama/llama-4-scout-17b-16e-instruct"]:
        if count == 0:
            domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df.pddl').read()

            problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf.pddl').read()

        else:
            domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_df_{count - 1}.pddl').read()
            problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model}/error_rag/{problem}/{problem}_{model}_pf_{count - 1}.pddl').read()
    else:
        if count == 0:
            domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df.pddl').read()

            problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf.pddl').read()

        else:
            domain_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
            problem_file = open(
                f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()

    plan_found = None

    req_body = {"domain": domain_file, "problem": problem_file}

    # Send job request to solve endpoint
    solve_request_url = requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve",
                                      json=req_body).json()

    # Query the result in the job
    celery_result = requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

    while celery_result.json().get("status", "") == 'PENDING':
        # Query the result every 0.5 seconds while the job is executing
        celery_result = requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])
        time.sleep(0.5)

    result = celery_result.json()['result']

    if "Error" in celery_result.json().keys():
        return False, "timeout"

    if solver == "lama-first":
        if not result['output']:
            if not result['stderr']:
                return False, result['stdout']
            else:
                return False, result['stderr']
        else:
            return True, result['output']
    elif solver == "dual-bfws-ffparser":
        if result['output'] == {'plan': ''}:
            if not result['stderr']:
                if "NOTFOUND" in result['stdout'] or "No plan" in result['stdout'] or "unknown" in result[
                    'stdout'] or "undeclared" in result['stdout'] or "declared twice" in result[
                    'stdout'] or "check input files" in result['stdout'] or "does not match" in result[
                    'stdout'] or "timeout" in result['call']:
                    plan_found = False
                else:
                    plan_found = True
                return plan_found, result['stdout']
            else:
                plan_found = False
                return plan_found, result['stderr']
        else:
            plan_found = True
            return plan_found, result['output']


def run_solver_batch(domain, model, data, index_start, index_end, solver):
    if '/' in model:
        _, model_name = model.split('/')
    else:
        model_name = model
    attempts = 3
    count = 4
    for problem in range(index_start, index_end):
        for n in range(count):
            # time.sleep(6)
            problem_name = "p0" + str(problem) if problem < 10 else "p" + str(problem)
            print(f"Running {problem_name}")

            for i in range(attempts):
                try:
                    plan_found, result = run_solver(domain, data, problem_name, model, solver, n)
                except:
                    if i < attempts - 1:
                        continue
                    else:
                        raise
                break

            if plan_found:
                plan_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem_name}_{prompt_version}/{problem_name}_{model_name}_plan.txt'
                # plan_path = f'../../output/llm-as-formalizer/blocksworld/Natural_BlocksWorld-111/{model}/pddl_instruction/{problem_name}/{problem_name}_plan.txt'
                if not os.path.exists(os.path.dirname(plan_path)):
                    os.makedirs(os.path.dirname(plan_path))
                if "Plan found with cost: 0" in result or "The empty plan solves it" in result:
                    plan = ''
                else:
                    plan = result['plan']
                with open(plan_path, 'w') as plan_file:
                    plan_file.write(plan)
                break

            else:
                # if n != count - 1:
                error_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/error_rag/{problem_name}_{prompt_version}/{problem_name}_{model_name}_error_{n}.txt'
                # error_path = f'../../output/llm-as-formalizer/blocksworld/Natural_BlocksWorld-111/R1-Qwen-32B/pddl_instruction/{problem_name}/{problem_name}_error.pddl'
                if not os.path.exists(os.path.dirname(error_path)):
                    os.makedirs(os.path.dirname(error_path))
                with open(error_path, 'w') as error_file:
                    error_file.write(result)

                # query = query_reformer(result,model)
                result = result.replace("\ndomain definition expected\n", "").replace("\n", "")


                if model in ["qwen/qwen3-8b","qwen/qwq-32b","qwen3-8b","qwq-32b"]:

                    for i in range(attempts):
                        try:
                            if "OK" in result or "ff" in result:
                                run_error_refine_qwen(domain, data, problem_name, model, result, n, doc=None)
                            else:
                                query = query_reformer_qwen(domain, data, problem_name, model, result, n, ).replace("`","").replace("pddl", "")

                                if "domain" in result:
                                    doc = search_documents(query, 2, "domain_data")
                                else:
                                    doc = search_documents(query, 2, "problem_data")

                                run_error_refine_qwen(domain, data, problem_name, model, result, n, query, doc)
                        except:
                            if i < attempts - 1:
                                continue
                            else:
                                raise
                        break

                else:

                    for i in range(attempts):
                        try:
                            if "OK" in result or "ff" in result:
                                run_error_refine_gpt(domain, data, problem_name, model, result, n, query, doc=None)

                            else:
                                query = query_reformer_gpt(domain, data, problem_name, model, result, n, ).replace("`",
                                                                                                                   "").replace(
                                    "pddl", "")

                                if "domain" in result:
                                    doc = search_documents(query, 2, "domain_data")
                                else:
                                    doc = search_documents(query, 2, "problem_data")

                                run_error_refine_gpt(domain, data, problem_name, model, result, n, query, doc)
                        except:
                            if i < attempts - 1:
                                continue
                            else:
                                raise
                        break


if __name__ == "__main__":
    args = Parser.parse_args()
    DOMAIN = args.domain
    MODEL = args.model
    DATA = args.data
    INDEX_START = eval(args.index_start)
    INDEX_END = eval(args.index_end)
    SOLVER = args.solver

    run_solver_batch(domain=DOMAIN, model=MODEL, data=DATA, index_start=INDEX_START, index_end=INDEX_END, solver=SOLVER)

