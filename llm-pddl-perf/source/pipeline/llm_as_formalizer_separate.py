from jsonlines import jsonlines  # type: ignore
from openai import OpenAI  # type: ignore
import json
import os
import time
import re

from kani import Kani  # type: ignore
from kani.engines.huggingface import HuggingEngine  # type: ignore
from kani.prompts.impl import LLAMA3_PIPELINE  # type: ignore
from kani.prompts.impl import GEMMA_PIPELINE  # type: ignore
import asyncio

import argparse

from transformers import AutoModelForCausalLM, BitsAndBytesConfig




quantization_config = BitsAndBytesConfig(load_in_8bit=True)

Parser = argparse.ArgumentParser()
Parser.add_argument("--domain", default="blocksworld", help="which domain to evaluate", choices=["blocksworld", "mystery_blocksworld", "barman", "logistics"])
Parser.add_argument("--model", default="deepseek/deepseek-r1-distill-llama-70b", help="which model to use", choices=["deepseek/deepseek-r1-distill-llama-70b","deepseek/deepseek-r1-distill-qwen-14b","qwen/qwen3-8b","qwen/qwq-32b","meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct","deepseek-r1-distill-qwen-14b","deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "qwen2.5-coder-7b-instruct", "qwq-32b", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-preview", "google/gemma-2-9b-it", "google/gemma-2-27b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "o3-mini", "meta-llama/Llama-3.1-405B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
Parser.add_argument("--data", default="Heavily_Templated_BlocksWorld-100", help="which data to formalize", choices=["Heavily_Templated_BlocksWorld-111", "Moderately_Templated_BlocksWorld-111", "Natural_BlocksWorld-111", "Heavily_Templated_Mystery_BlocksWorld-100", "Heavily_Templated_Barman-100", "Heavily_Templated_Logistics-100", "Moderately_Templated_Logistics-100", "Natural_Logistics-100", "Heavily_Templated_BlocksWorld-100"])
Parser.add_argument("--index_start", default="55", help="index to start generating result from (inclusive)")
Parser.add_argument("--index_end", default="101", help="index to end generating result from (exclusive)")

args = Parser.parse_args()
DOMAIN = args.domain
MODEL = args.model
DATA = args.data
INDEX_START = eval(args.index_start)
INDEX_END = eval(args.index_end)

OPEN_SOURCED_MODELS = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it",
                       "meta-llama/Llama-3.1-70B-Instruct", "google/gemma-2-27b-it",
                       "meta-llama/Llama-3.1-405B-Instruct", "meta-llama/Llama-3.3-70B-Instruct",
                       "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                       "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
PROMPT = "You are a Planning Domain Definition Language (PDDL) expert."
if MODEL in OPEN_SOURCED_MODELS:
    if "Llama-3.1-405B-Instruct" in MODEL:
        ENGINE = HuggingEngine(model_id=MODEL, prompt_pipeline=LLAMA3_PIPELINE, use_auth_token=True,
                               max_new_tokens=10000,
                               model_load_kwargs={"device_map": "auto", "quantization_config": quantization_config})
    elif "gemma" in MODEL:
        ENGINE = HuggingEngine(model_id=MODEL, prompt_pipeline=GEMMA_PIPELINE, use_auth_token=True,
                               max_new_tokens=10000)
    elif "deepseek-ai" in MODEL:
        ENGINE = HuggingEngine(model_id=MODEL, prompt_pipeline=None, use_auth_token=True, max_new_tokens=10000)
    AI = Kani(ENGINE, system_prompt=PROMPT)
else:
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


# def run_formalizer_gpt(domain, data, problem, model, force_json=False):
#     output_format = "json_object" if force_json else "text"
#     continue_list = ["domain definition", "requirement", "predicates", "action"]
#     domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt',encoding="utf-8").read()
#     problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt').read()
#
#     prompt = f'''You are a PDDL expert.
# Is next step needed here: yes
# next step: domain definition
# domain file:
# (define (domain construction)
# )
#
# Is next step needed here: yes
# next step: requirements
# domain file:
# (define (domain game)
#   (:requirements :strips :adl :typing)
# )
#
# Is next step needed here: yes
# next step: predicates
# domain file:
# (define (domain game)
#   (:requirements :strips)
#   (:predicates
#     (walls-built ?s - site)
#     (windows-fitted ?s - site)
#     (foundations-set ?s - site)
#     (cables-installed ?s - site)
#     (site-built ?s - site)
#     (on-site ?m - material ?s - site)
#     (material-used ?m - material)
#   )
# )
#
# Is next step needed here: yes
# next step: action
# domain file:
# (define (domain game)
#   (:requirements :strips)
#   (:predicates
#     (province ?o)
#     (planet ?o)
#     (harmony)
#     (pain ?o)
#     (craves ?o1 ?o2)
#   )
#   (:action BUILD-WALL
#   :parameters (?s - site ?b - bricks)
#   :precondition (and
#     (on-site ?b ?s)
#     (foundations-set ?s)
#     (not (walls-built ?s))
#     (not (material-used ?b))
#   )
#   :effect (and
#     (walls-built ?s)
#     (material-used ?b)
#   )
#   )
# )
#
#
#
# follow the example above, generate PDDL step by step
# Here is a game we are playing.
# {domain_description}
#
# '''
#
#
#
#
#     next_step = "no"
#     i = 0
#     while next_step != "action" and next_step != "actions" and i < 6:
#         # next = continue_list[i]
#
#         # doc = search_documents(next,2,"domain_data")
#
#         # doc = "\n\n".join(doc)
#
#         if i != 0:
#
#             prompt = prompt + f'''
# Is next step needed here: {is_needed}
# next step: {next_step}
# domain file:
# {PDDL}
# '''
#
#         i = i + 1
#         message = prompt + "Return a JSON object in the following format:\n{\n  \"Is_next_step_needed_here\": \"...\",\n  \"next_step\": \"...\",\n  \"PDDL\": \"...\"\n}"
#
#         print(message)
#
#         completion = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": message}
#             ],
#             response_format={
#                 'type': 'text'
#             }
#         )
#
#         return_string = completion.choices[0].message.content
#
#         # print(return_string)
#
#         if model in ['o1-preview', 'deepseek-reasoner', 'qwen2.5-coder-7b-instruct']:
#             start_index = return_string.find('{')
#             end_index = return_string.find('}')
#             json_string = return_string[start_index:end_index + 1]
#             print(json_string)
#             return_dict = json.loads(json_string, strict=False)
#         else:
#             return_dict = json.loads(return_string, strict=False)
#
#         is_needed = return_dict["Is_next_step_needed_here"]
#         next_step = return_dict["next_step"]
#         PDDL = return_dict["PDDL"]
#         domain_file = PDDL
#         # print(current)
#
#
#
# #     prompt_problem = f'''You are a PDDL expert. Here is a game we are playing.
# # {domain_file}
# # {problem_description}
# # Read the domain file carefully then write a minimal corresponding problem file.
# # '''
#     prompt_problem = f'''You are a PDDL expert.
# Is next step needed here: yes
# next step: problem definition
# problem file:
# (define (problem block-stacking-problem)
#   (:domain domain game)
# )
#
# Is next step needed here: yes
# next step: objects
# problem file:
# (define (problem block-stacking-problem)
#   (:domain domain game)
#   (:objects
#     s1 s2 s3 - site
#     ba bb bc - bricks
#     windowAlpha windowBeta windowCharlie - windows
#     uierjiae faufhf uihrai - cables
#   )
# )
#
# Is next step needed here: yes
# next step: Init
# problem file:
# (define (problem block-stacking-problem)
#   (:domain domain game)
#   (:objects
#     s1 s2 s3 - site
#     ba bb bc - bricks
#     windowAlpha windowBeta windowCharlie - windows
#     uierjiae faufhf uihrai - cables
#   )
#   (:init
#     (on-site b s1)
#     (on-site c s1)
#     (on-site w s1)
#   )
# )
#
# Is next step needed here: yes
# next step: Goal
# problem file:
# (define (problem block-stacking-problem)
#   (:domain domain game)
#   (:objects
#     s1 s2 s3 - site
#     ba bb bc - bricks
#     windowAlpha windowBeta windowCharlie - windows
#     uierjiae faufhf uihrai - cables
#   )
#   (:init
#     (on-site b s1)
#     (on-site c s1)
#     (on-site w s1)
#   )
#   (:goal (and
#   (walls-built ?s1)
#   (cables-installed ?s1)
#   (windows-fitted ?s1)
#   )
#   )
# )
#
# Following the example above.
# Here is a game we are playing. Base on the domain file and problem_description, generate problem file step by step.
# domain file
# {domain_file}
# problem_description:
# {problem_description}
#
# '''
#
#
#     next_step = "no"
#     i = 0
#     while next_step != "Goal" and next_step != "goal" and i < 6:
#         # next = continue_list[i]
#
#         # doc = search_documents(next,2,"domain_data")
#
#         # doc = "\n\n".join(doc)
#
#         if i != 0:
#             prompt = prompt + f'''
# Is next step needed here: {is_needed}
# next step: {next_step}
# problem file:
# {PDDL}
#     '''
#
#         i = i + 1
#         message_problem = prompt_problem + "Return a JSON object in the following format:\n{\n  \"Is_next_step_needed_here\": \"...\",\n  \"next_step\": \"...\",\n  \"PDDL\": \"...\"\n}"
#
#         print(message)
#
#         completion = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": message_problem}
#             ],
#             response_format={
#                 'type': 'text'
#             }
#         )
#
#         return_string = completion.choices[0].message.content
#
#         # print(return_string)
#
#         if model in ['o1-preview', 'deepseek-reasoner', 'qwen2.5-coder-7b-instruct']:
#             start_index = return_string.find('{')
#             end_index = return_string.find('}')
#             json_string = return_string[start_index:end_index + 1]
#             print(json_string)
#             return_dict = json.loads(json_string, strict=False)
#         else:
#             return_dict = json.loads(return_string, strict=False)
#
#         is_needed = return_dict["Is_next_step_needed_here"]
#         next_step = return_dict["next_step"]
#         PDDL = return_dict["PDDL"]
#         problem_file = PDDL
#
#     df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_df.pddl'
#     pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_pf.pddl'
#
#     if not os.path.exists(os.path.dirname(df_path)):
#         os.makedirs(os.path.dirname(df_path))
#
#     with open(df_path, 'w') as df:
#         df.write(domain_file)
#
#     with open(pf_path, 'w') as pf:
#         pf.write(problem_file)
#
#     return domain_file, problem_file


def run_formalizer_gpt(domain, data, problem, model, force_json=False):

    if "meta" in model or "google" in model or "deepseek-ai" in model:
        _, model_name = model.split("/")
    else:
        _, model_name = model.split("/")
    output_format = "json_object" if force_json else "text"
    continue_list = ["domain definition", "requirement", "predicates", "action"]
    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt',encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt').read()

    prompt = f'''You are a PDDL expert.
Is next step needed here: yes
next step: domain definition
next domain file:
(define (domain construction)
)

Is next step needed here: yes
next step: requirements
next domain file:
(define (domain game)
  (:requirements :strips :adl :typing)
)

Is next step needed here: yes
next step: predicates
next domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates
    (walls-built ?s - site)
    (windows-fitted ?s - site)
    (foundations-set ?s - site)
    (cables-installed ?s - site)
    (site-built ?s - site)
    (on-site ?m - material ?s - site)
    (material-used ?m - material)
  )
)

Is next step needed here: yes
next step: actions
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)

follow the example above, please do not add any other word, generate only the next one step for the domain file.
Here is a game we are playing.
{domain_description}
{problem_description}

'''

    prompt_problem = f'''You are a PDDL expert.
Is next step needed here: yes
next step: domain definition
domain file:
(define (domain construction)
)

Is next step needed here: yes
next step: requirements
domain file:
(define (domain game)
  (:requirements :strips :adl :typing)
)

Is next step needed here: yes
next step: predicates
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates
    (walls-built ?s - site)
    (windows-fitted ?s - site)
    (foundations-set ?s - site)
    (cables-installed ?s - site)
    (site-built ?s - site)
    (on-site ?m - material ?s - site)
    (material-used ?m - material)
  )
)

Is next step needed here: yes
next step: action
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)

Is next step needed here: yes
next step: problem file
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)
problem file:
(define (problem block-stacking-problem)
  (:domain domain game)
  (:objects
    s1 s2 s3 - site
    ba bb bc - bricks
    windowAlpha windowBeta windowCharlie - windows
    uierjiae faufhf uihrai - cables
  )
  (:init
    (on-site b s1)
    (on-site c s1)
    (on-site w s1)
  )
)

follow the example above, please do not add any other word, generate problem file only.
Here is a game we are playing.
{domain_description}
{problem_description}

'''


    next_step = "no"
    i = 0
    while next_step != "action" and next_step != "actions" and i < 6:
        # next = continue_list[i]

        # doc = search_documents(next,2,"domain_data")

        # doc = "\n\n".join(doc)

        if i != 0:

            prompt = prompt + f'''
Is next step needed here: {is_needed}
next step: {next_step}
domain file:
{PDDL}
'''
            prompt_problem = prompt_problem + f'''
Is next step needed here: {is_needed}
next step: {next_step}
domain file:
{PDDL}
'''
        i = i + 1
        message = prompt + "Return a JSON object in the following format:\n{\n  \"Is_next_step_needed_here\": \"...\",\n  \"next_step\": \"...\",\n  \"next_domain_file\": \"...\"\n}"

        print(message)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
            response_format={
                'type': "json_object"
            }
        )

        return_string = completion.choices[0].message.content

        # print(return_string)

        if model in ["deepseek/deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-qwen-14b", "qwen/qwen3-8b",
                     "qwen/qwq-32b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                     "meta-llama/llama-4-scout-17b-16e-instruct",
                     'o1-preview', 'deepseek-reasoner', "deepseek-r1-distill-llama-70b", "deepseek-r1-distill-qwen-7b",
                     "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-14b", "qwen2.5-coder-7b-instruct",
                     "qwen3-8b", "deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct",
                     "deepseek-r1-distill-llama-70b"]:
            start_index = return_string.find('{')
            end_index = return_string.rfind('}')
            json_string = return_string[start_index:end_index + 1]
            print(json_string)
            return_dict = json.loads(json_string, strict=False)
        else:
            return_dict = json.loads(return_string, strict=False)

        is_needed = return_dict["Is_next_step_needed_here"]
        next_step = return_dict["next_step"]
        PDDL = return_dict["next_domain_file"]
        domain_file = PDDL
        # print(current)



#     prompt_problem = f'''You are a PDDL expert. Here is a game we are playing.
# {domain_file}
# {problem_description}
# Read the domain file carefully then write a minimal corresponding problem file.
# '''
    message_problem = prompt_problem + "Return a JSON object in the following format:\n{\n  \"problem file\":...\n}"

    print(message_problem)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": message_problem}
        ],
        response_format={
            'type': "json_object"
        }
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
        end_index = return_string.rfind('}')
        json_string = return_string[start_index:end_index + 1]
        print(json_string)
        return_dict = json.loads(json_string, strict=False)
    else:
        return_dict = json.loads(return_string, strict=False)

    problem_file = return_dict["problem file"]

    if model in ["meta-llama/llama-4-maverick-17b-128e-instruct","meta-llama/llama-4-scout-17b-16e-instruct"]:
        df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_df.pddl'
        pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_pf.pddl'
    else:
        df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/separate/{problem}/{problem}_{model_name}_df.pddl'
        pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/separate/{problem}/{problem}_{model_name}_pf.pddl'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file




def run_formalizer_qwen(domain, data, problem, model, force_json=False):
    output_format = "json_object" if force_json else "text"
    continue_list = ["domain definition", "requirement", "predicates", "action"]
    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt',encoding="utf-8").read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt').read()

    prompt = f'''You are a PDDL expert.
Is next step needed here: yes
next step: domain definition
next domain file:
(define (domain construction)
)

Is next step needed here: yes
next step: requirements
next domain file:
(define (domain game)
  (:requirements :strips :adl :typing)
)

Is next step needed here: yes
next step: predicates
next domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates
    (walls-built ?s - site)
    (windows-fitted ?s - site)
    (foundations-set ?s - site)
    (cables-installed ?s - site)
    (site-built ?s - site)
    (on-site ?m - material ?s - site)
    (material-used ?m - material)
  )
)

Is next step needed here: yes
next step: actions
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)

follow the example above, generate domain file step by step, only generate next one step.
Here is a game we are playing.
{domain_description}
{problem_description}

'''

    prompt_problem = f'''You are a PDDL expert.
Is next step needed here: yes
next step: domain definition
domain file:
(define (domain construction)
)

Is next step needed here: yes
next step: requirements
domain file:
(define (domain game)
  (:requirements :strips :adl :typing)
)

Is next step needed here: yes
next step: predicates
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates
    (walls-built ?s - site)
    (windows-fitted ?s - site)
    (foundations-set ?s - site)
    (cables-installed ?s - site)
    (site-built ?s - site)
    (on-site ?m - material ?s - site)
    (material-used ?m - material)
  )
)

Is next step needed here: yes
next step: action
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)

Is next step needed here: yes
next step: problem file
domain file:
(define (domain game)
  (:requirements :strips)
  (:predicates 
    (province ?o)
    (planet ?o)
    (harmony)
    (pain ?o)
    (craves ?o1 ?o2)
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
  (:action BUILD-WALL
  :parameters (?s - site ?b - bricks)
  :precondition (and
    (on-site ?b ?s)
    (foundations-set ?s)
    (not (walls-built ?s))
    (not (material-used ?b))
  )
  :effect (and
    (walls-built ?s)
    (material-used ?b)
  )
  )
)
problem file:
(define (problem block-stacking-problem)
  (:domain domain game)
  (:objects
    s1 s2 s3 - site
    ba bb bc - bricks
    windowAlpha windowBeta windowCharlie - windows
    uierjiae faufhf uihrai - cables
  )
  (:init
    (on-site b s1)
    (on-site c s1)
    (on-site w s1)
  )
)

follow the example above, please do not add any other word, generate problem file only.
Here is a game we are playing.
{domain_description}
{problem_description}

'''

    next_step = "no"
    i = 0
    while next_step != "action" and next_step != "actions" and i < 6:
        # next = continue_list[i]

        # doc = search_documents(next,2,"domain_data")

        # doc = "\n\n".join(doc)

        if i != 0:

            prompt = prompt + f'''
Is next step needed here: {is_needed}
next step: {next_step}
domain file: {PDDL}
'''
            prompt_problem = prompt_problem + f'''
Is next step needed here: {is_needed}
next step: {next_step}
domain file: {PDDL}
'''
        i = i + 1
        message_problem = prompt_problem + "Return a JSON object in the following format:\n{\n  \"Is_next_step_needed_here\": \"...\",\n  \"next_step\": \"...\",\n  \"PDDL\": \"...\"\n}"

        print(message_problem)

        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""  # 定义完整回复
        is_answering = False  # 判断是否结束思考过程并开始回复

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=model,  # 此处以 qwq-32b 为例，可按需更换模型名称
            messages=[
                {"role": "user", "content": message_problem}
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

        # print(return_string)

        if model in ['o1-preview', 'deepseek-reasoner', 'qwen2.5-coder-7b-instruct', 'qwq-32b']:
            start_index = return_string.find('{')
            end_index = return_string.find('}')
            json_string = return_string[start_index:end_index + 1]
            print(json_string)
            return_dict = json.loads(json_string, strict=False)
        else:
            return_dict = json.loads(return_string, strict=False)

        is_needed = return_dict["Is_next_step_needed_here"]
        next_step = return_dict["next_step"]
        PDDL = return_dict["PDDL"]
        domain_file = PDDL
        # print(current)



#     prompt_problem = f'''You are a PDDL expert. Here is a game we are playing.
# {domain_file}
# {problem_description}
# Read the domain file carefully then write a minimal corresponding problem file.
# '''
    message_problem = prompt_problem + "Return a JSON object in the following format:\n{\n  \"problem file\":...\n}"

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model=model,  # 此处以 qwq-32b 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": message_problem}
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

    # print(return_string)

    if model in ['o1-preview', 'deepseek-reasoner', 'qwen2.5-coder-7b-instruct']:
        start_index = return_string.find('{')
        end_index = return_string.find('}')
        json_string = return_string[start_index:end_index + 1]
        print(json_string)
        return_dict = json.loads(json_string, strict=False)
    else:
        return_dict = json.loads(return_string, strict=False)

    problem_file = return_dict["problem file"]

    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_df.pddl'
    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model}/separate/{problem}/{problem}_{model}_pf.pddl'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    return domain_file, problem_file




async def run_formalizer_open_sourced(domain, data, problem):
    domain_description = open(f'../../data/textual_{domain}/{data}/{problem}_domain.txt').read()
    problem_description = open(f'../../data/textual_{domain}/{data}/{problem}_problem.txt').read()

    prompt_prefix = open(f'../../prompts/pddl_instruction.txt').read()
    prompt_version = "pddl_instruction"
    message = prompt_prefix + f"""The following is the domain description and problem description in naturalistic language. Domain description:\n{domain_description}\n Problem description: {problem_description}
    Write the domain and problem files in Planning Domain Definition Language (PDDL).
    The domain file should be inside the tags <domain_file> ... </domain_file> as in the above example.
    The problem file should be inside the tags <problem_file> ... </problem_file> as in the above example."""
    response = await AI.chat_round_str(message)

    # try:
    #     _, domain_file, _, problem_file, _ = response.split('```')
    # except:
    #     domain_file_index = response.index("(define (domain") if "(define (domain" in response else 0
    #     problem_file_index = response.index("(define (problem ") if "(define (problem " in response else 0
    #     domain_file = response[ domain_file_index: problem_file_index].strip()
    #     problem_file = response[problem_file_index : ].strip()

    full_response = response[:]
    response = response.split("</think>", 1)[-1]
    domain_pattern = re.search(r"<domain_file>(.*?)</domain_file>", response, re.DOTALL)
    problem_pattern = re.search(r"<problem_file>(.*?)</problem_file>", response, re.DOTALL)

    domain_file = domain_pattern.group(1).strip() if domain_pattern else None
    problem_file = problem_pattern.group(1).strip() if problem_pattern else None

    # domain_file = domain_file.replace("pddl", "").replace("lisp", "")
    # problem_file = problem_file.replace("pddl", "").replace("lisp", "")

    _, model_name = MODEL.split('/')
    df_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{prompt_version}/{problem}/{problem}_{model_name}_df.pddl'
    pf_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{prompt_version}/{problem}/{problem}_{model_name}_pf.pddl'
    res_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{prompt_version}/{problem}/{problem}_{model_name}_response.md'

    if not os.path.exists(os.path.dirname(df_path)):
        os.makedirs(os.path.dirname(df_path))

    with open(df_path, 'w') as df:
        df.write(domain_file)

    with open(pf_path, 'w') as pf:
        pf.write(problem_file)

    with open(res_path, 'w') as res:
        res.write(full_response)

    return domain_file, problem_file


def run_gpt_batch(domain, model, data, index_start, index_end):
    for problem_number in range(index_start, index_end):
        problem_name = 'p0' + str(problem_number) if problem_number < 10 else 'p' + str(problem_number)
        force_json = True if model not in ["o1-preview", "deepseek-reasoner", "qwen2.5-coder-7b-instruct"] else False
        print(f"Running {problem_name}")
        run_formalizer_gpt(domain=domain, data=data, problem=problem_name, model=model, force_json=force_json)

def run_qwen_batch(domain, model, data, index_start, index_end):
    for problem_number in range(index_start, index_end):
        problem_name = 'p0' + str(problem_number) if problem_number < 10 else 'p' + str(problem_number)
        force_json = True if model not in ["o1-preview", "deepseek-reasoner", "qwen2.5-coder-7b-instruct"] else False
        print(f"Running {problem_name}")
        run_formalizer_qwen(domain=domain, data=data, problem=problem_name, model=model, force_json=force_json)
async def run_open_sourced_batch(domain, data, index_start, index_end):
    problem_names = ['p0' + str(problem) if problem < 10 else 'p' + str(problem) for problem in
                     range(index_start, index_end)]
    tasks = [run_formalizer_open_sourced(domain, data, problem) for problem in problem_names]
    outputs = await asyncio.gather(*tasks)
    return outputs


if __name__ == "__main__":
    start_time = time.time()

    if MODEL in OPEN_SOURCED_MODELS:
        asyncio.run(run_open_sourced_batch(domain=DOMAIN, data=DATA, index_start=INDEX_START, index_end=INDEX_END))
    elif MODEL in ["qwen3-8b", "qwq-32b"]:
        run_qwen_batch(domain=DOMAIN, model=MODEL, data=DATA, index_start=INDEX_START, index_end=INDEX_END)
    else:
        run_gpt_batch(domain=DOMAIN, model=MODEL, data=DATA, index_start=INDEX_START, index_end=INDEX_END)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the script: {elapsed_time:.2f} seconds")