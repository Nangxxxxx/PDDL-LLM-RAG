import requests
import time
import pandas as pd
import os
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument("--domain", default="mystery_blocksworld", help="which domain to evaluate", choices=["blocksworld", "mystery_blocksworld", "barman", "logistics"])
Parser.add_argument("--model", default="deepseek/deepseek-r1-distill-llama-70b", help="which model to use", choices=["deepseek/deepseek-r1-distill-llama-70b","deepseek/deepseek-r1-distill-qwen-14b","qwen/qwen3-8b","qwen/qwq-32b","meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct","deepseek-r1-distill-qwen-14b","deepseek-r1-distill-llama-70b", "llama-4-scout-17b-16e-instruct", "qwen3-8b", "llama-4-maverick-17b-128e-instruct", "qwen2.5-coder-7b-instruct", "qwq-32b", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-preview", "google/gemma-2-9b-it", "google/gemma-2-27b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "o3-mini", "meta-llama/Llama-3.1-405B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"])
Parser.add_argument("--data", default="Heavily_Templated_Mystery_BlocksWorld-100", help="which data to use", choices=["Heavily_Templated_BlocksWorld-111", "Moderately_Templated_BlocksWorld-111", "Natural_BlocksWorld-111", "Heavily_Templated_Mystery_BlocksWorld-100", "Heavily_Templated_Barman-100", "Heavily_Templated_Logistics-100", "Moderately_Templated_Logistics-100", "Natural_Logistics-100", "Heavily_Templated_BlocksWorld-100"])
Parser.add_argument("--index_start", default="1", help="index to start generating result from (inclusive)")
Parser.add_argument("--index_end", default="32", help="index to end generating result from (exclusive)")
Parser.add_argument("--solver", help="which solver to use", default="dual-bfws-ffparser")
Parser.add_argument("--type", default="separate", choices=["rag","formalize","steady"])

prompt_version = "pddl_instruction"


def run_solver(domain, data, problem, model, solver, type):
    if "meta" in model or "google" in model or "deepseek" in model:
        name, model_name = model.split("/")
    else:
        model_name = model

    # domain_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_df.pddl').read()
    # problem_file = open(f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{problem}_{prompt_version}/{problem}_{model_name}_pf.pddl').read()

    domain_file = open(
        # f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem}/{problem}_{name}/{model_name}_df.pddl').read()
        f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem}/{problem}_{model_name}_df.pddl').read()
    problem_file = open(
        # f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem}/{problem}_{name}/{model_name}_pf.pddl').read()
        f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem}/{problem}_{model_name}_pf.pddl').read()

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


def run_solver_batch(domain, model, data, index_start, index_end, solver, type):
    if '/' in model:
        _, model_name = model.split('/')
    else:
        model_name = model
    attempts = 3
    for problem in range(index_start, index_end):
        problem_name = "p0" + str(problem) if problem < 10 else "p" + str(problem)
        print(f"Running {problem_name}")
        for i in range(attempts):
            try:
                plan_found, result = run_solver(domain, data, problem_name, model, solver, type)
            except:
                if i < attempts - 1:
                    continue
                else:
                    raise
            break
        if plan_found:
            plan_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem_name}_{prompt_version}/{problem_name}_{model_name}_plan.txt'
            # plan_path = f'../../output/llm-as-formalizer/blocksworld/{data}/{model}/{prompt_version}/{problem_name}/{problem_name}_plan.txt'
            if not os.path.exists(os.path.dirname(plan_path)):
                os.makedirs(os.path.dirname(plan_path))
            if "Plan found with cost: 0" in result or "The empty plan solves it" in result:
                plan = ''
            else:
                plan = result['plan']
            with open(plan_path, 'w') as plan_file:
                plan_file.write(plan)
        else:
            error_path = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{type}/{problem_name}_{prompt_version}/{problem_name}_{model_name}_error.txt'
            # error_path = f'../../output/llm-as-formalizer/blocksworld/{data}/{model}/{prompt_version}/{problem_name}/{problem_name}_error.pddl'
            if not os.path.exists(os.path.dirname(error_path)):
                os.makedirs(os.path.dirname(error_path))
            with open(error_path, 'w') as error_file:
                error_file.write(result)


if __name__ == "__main__":
    args = Parser.parse_args()
    DOMAIN = args.domain
    MODEL = args.model
    DATA = args.data
    INDEX_START = eval(args.index_start)
    INDEX_END = eval(args.index_end)
    SOLVER = args.solver
    TYPE = args.type

    run_solver_batch(domain=DOMAIN, model=MODEL, data=DATA, index_start=INDEX_START, index_end=INDEX_END, solver=SOLVER, type=TYPE)

