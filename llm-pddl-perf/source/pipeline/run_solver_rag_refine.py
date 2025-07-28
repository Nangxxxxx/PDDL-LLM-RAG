import json

import requests
import time
import pandas as pd
import os
import argparse
from jsonlines import jsonlines
from openai import OpenAI


from elasticsearch import Elasticsearch, NotFoundError



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
                    "fields": ["type_name", "document", "example"],  # 查询 title 和 content 字段
                    "boost": 2.0  # 为 title 字段增加权重
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
            combined_text = f"type_name: {type_name}\ndocumentation: {documentation}\nexample: {example}"  # 将 title 和 content 组合成一个字符串
            combined_list.append(combined_text)

    return combined_list

create_index("domain_data")  # 创建索引
create_index("problem_data")  # 创建索引
add_documents("domain_data")  # 添加文档
add_documents("problem_data")  # 添加文档



Parser = argparse.ArgumentParser()
Parser.add_argument("--domain", default="logistics", help="which domain to evaluate", choices=["blocksworld", "mystery_blocksworld", "barman", "logistics"])
Parser.add_argument("--model", default="deepseek/deepseek-r1-distill-llama-70b", help="which model to use", choices=["deepseek/deepseek-r1-distill-llama-70b","deepseek/deepseek-r1-distill-qwen-14b","qwen/qwen3-8b","qwen/qwq-32b","meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct","deepseek-r1-distill-qwen-14b","deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "qwen2.5-coder-7b-instruct", "qwq-32b", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-preview", "google/gemma-2-9b-it", "google/gemma-2-27b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "o3-mini", "meta-llama/Llama-3.1-405B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
Parser.add_argument("--data", default="Heavily_Templated_Logistics-100", help="which data to use", choices=["Moderately_Templated_BlocksWorld-100", "Heavily_Templated_BlocksWorld-111", "Moderately_Templated_BlocksWorld-111", "Natural_BlocksWorld-111", "Heavily_Templated_Mystery_BlocksWorld-100", "Heavily_Templated_Barman-100", "Heavily_Templated_Logistics-100", "Moderately_Templated_Logistics-100", "Natural_Logistics-100", "Heavily_Templated_BlocksWorld-100"])
Parser.add_argument("--index_start", default="33", help="index to start generating result from (inclusive)")
Parser.add_argument("--index_end", default="51", help="index to end generating result from (exclusive)")
Parser.add_argument("--solver", help="which solver to use", default="dual-bfws-ffparser")

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
# def query_reformer(doc, model, force_json=False):
#     output_format = "json_object" if force_json else "text"
#
#     prompt = f"error:\n{doc}\n\nInstruction: I provided a pddl error. Please rewrite it into a short keyword query to facilitate my search. Please give the query directly."
#
#     # message = prompt + "Return a JSON object in the following format:\n{\n  \"rewrited_doc\": ...\n}"
#
#     completion = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             response_format={"type": output_format}
#         )
#
#     return_string = completion.choices[0].message.content
#     print(return_string)
#     return return_string



def run_error_refine_gpt(domain, data, problem, model, result, count, doc=None, force_json=False):
    if "/" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        model_name = model

    output_format = "json_object" if force_json else "text"

    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt',encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt',encoding="utf-8").read()
    
    if count == 0:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf.pddl').read()
    else:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df_{count-1}.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf_{count-1}.pddl').read()

    doc = "\n\n".join(doc)
    #
    prompt = f"Knowledge:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"domain_description:\n{domain_description}\n\nproblem_description:\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\ndoc and example:\n{doc}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback and doc and example, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"Knowledge:\n{doc}\n\n\nYou are a PDDL expert. Here is a game we are playing.\n{domain_description}\n{problem_description}\nWrite the domain and problem files in minimal PDDL."


    message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n  \"reason\":...\n}"

    print(message)


    completion = client.chat.completions.create(
        model=model,

        messages=[
            {"role": "system", "content": message},
            {"role": "user", "content": message}
        ],
        # response_format = {
        #     'type': 'text'
        # }
    )

    return_string = completion.choices[0].message.content

    print(return_string)

    if model in ["deepseek/deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-qwen-14b", "qwen/qwen3-8b",
                 "qwen/qwq-32b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                 "meta-llama/llama-4-scout-17b-16e-instruct",
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

    domain_file = return_dict["corrected domain file"]
    problem_file = return_dict["corrected problem file"]


    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/rag_refine/{problem}/{problem}_{model}_df_{count}.pddl'

    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/rag_refine/{problem}/{problem}_{model}_pf_{count}.pddl'


    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file


def run_error_refine_qwen(domain, data, problem, model, result, count, doc=None, force_json=False):
    if "/" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        model_name = model
    output_format = "json_object" if force_json else "text"

    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt', encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt', encoding="utf-8").read()

    if count == 0:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf.pddl').read()
    else:
        previous_domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df_{count - 1}.pddl').read()
        previous_problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf_{count - 1}.pddl').read()

    doc = "\n\n".join(doc)
    #
    prompt = f"Knowledge:\n{doc}\n\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"domain_description:\n{domain_description}\n\nproblem_description:\n{problem_description}\n\nWrong_domain_file:\n{previous_domain_file}\n\nWrong_problem_file:\n{previous_problem_file}\n\nerror feedback:{result}\n\ndoc and example:\n{doc}\n\nInstruction: I provided a wrong set of PDDL files, you need according to the error feedback and doc and example, give me the corrected domain_file and problem_file. You must make changes, and give me a logical reason for why you change like that.\n"
    # prompt = f"Knowledge:\n{doc}\n\n\nYou are a PDDL expert. Here is a game we are playing.\n{domain_description}\n{problem_description}\nWrite the domain and problem files in minimal PDDL."

    message = prompt + "Return a JSON object in the following format:\n{\n  \"corrected domain file\": ...,\n  \"corrected problem file\":...,\n  \"reason\":...\n}"

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
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

    print(return_string)

    if model in ["deepseek/deepseek-r1-distill-qwen-14b", "qwen/qwen3-8b","qwen/qwq-32b", 'o1-preview', 'deepseek-reasoner', "qwen2.5-coder-7b-instruct", "qwq-32b", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "llama-4-scout-17b-16e-instruct"]:
        start_index = return_string.find('{')
        end_index = return_string.find('}')
        json_string = return_string[start_index:end_index + 1]
        return_dict = json.loads(json_string, strict=False)
    else:
        return_dict = json.loads(return_string, strict=False)

    domain_file = return_dict["corrected domain file"]
    problem_file = return_dict["corrected problem file"]

    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df_{count}.pddl'

    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf_{count}.pddl'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file


def run_solver(domain, data, problem, model, solver, count):
    if "/" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        model_name = model

    # domain_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_df.pddl').read()
    # problem_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_pf.pddl').read()


    if count == 0:
        domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df.pddl').read()

        problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf.pddl').read()

    else:
        domain_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_df_{count-1}.pddl').read()
        problem_file = open(
            f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem}/{problem}_{model_name}_pf_{count-1}.pddl').read()




    plan_found = None


    req_body = {"domain" : domain_file, "problem" : problem_file}

    # Send job request to solve endpoint
    solve_request_url=requests.post(f"https://solver.planning.domains:5001/package/{solver}/solve", json=req_body).json()

    # Query the result in the job
    celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])

    while celery_result.json().get("status","")== 'PENDING':
        # Query the result every 0.5 seconds while the job is executing
        celery_result=requests.post('https://solver.planning.domains:5001' + solve_request_url['result'])
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
                if "NOTFOUND" in result['stdout'] or "No plan" in result['stdout'] or "unknown" in result['stdout'] or "undeclared" in result['stdout'] or "declared twice" in result['stdout'] or "check input files" in result['stdout'] or "does not match" in result['stdout'] or "timeout" in result['call']:
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
        corpus = []
        for n in range(count):
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
                plan_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem_name}_{prompt_version}/{problem_name}_{model_name}_plan.txt'
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
                error_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/rag_refine/{problem_name}_{prompt_version}/{problem_name}_{model_name}_error_{n}.txt'
                # error_path = f'../../output/llm-as-formalizer/blocksworld/Natural_BlocksWorld-111/R1-Qwen-32B/pddl_instruction/{problem_name}/{problem_name}_error.pddl'
                if not os.path.exists(os.path.dirname(error_path)):
                    os.makedirs(os.path.dirname(error_path))
                with open(error_path, 'w') as error_file:
                    error_file.write(result)

                # query = query_reformer(result,model)
                if "domain" in result:
                    doc = search_documents(result,3, "domain_data")
                else:
                    doc = search_documents(result,3, "problem_data")

                for line in doc:
                    corpus.append(line)
                corpus = list(set(corpus))

                if "qwen" in model or "qwq" in model:
                    for i in range(attempts):
                        try:
                            run_error_refine_qwen(domain, data, problem_name, model, result, n, corpus)
                        except:
                            if i < attempts - 1:
                                continue
                            else:
                                raise
                        break

                else:
                    for i in range(attempts):
                        try:
                            run_error_refine_gpt(domain, data, problem_name, model, result, n, corpus)
                        except:
                            if i < attempts - 1:
                                continue
                            else:
                                raise
                        break


if __name__=="__main__":
    args = Parser.parse_args()
    DOMAIN = args.domain
    MODEL = args.model
    DATA = args.data
    INDEX_START = eval(args.index_start)
    INDEX_END = eval(args.index_end)
    SOLVER = args.solver

    run_solver_batch(domain=DOMAIN, model=MODEL, data=DATA, index_start=INDEX_START, index_end=INDEX_END, solver=SOLVER)

