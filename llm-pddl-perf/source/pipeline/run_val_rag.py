import os
import subprocess
import pandas as pd
import re
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument("--domain", default="barman", help="which domain to evaluate",choices=["blocksworld", "mystery_blocksworld", "barman", "logistics"])
Parser.add_argument("--model", default="qwq-32b", help="which model to use",choices=["llama-4-maverick-17b-128e-instruct","gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o1-preview", "google/gemma-2-9b-it","google/gemma-2-27b-it", "meta-llama/Meta-Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct", "o3-mini", "meta-llama/Llama-3.1-405B-Instruct","deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B","deepseek-chat"])
Parser.add_argument("--data", default="Heavily_Templated_Barman-100", help="which data to validate",choices=["Heavily_Templated_BlocksWorld-111", "Moderately_Templated_BlocksWorld-111","Natural_BlocksWorld-111", "Heavily_Templated_Mystery_BlocksWorld-100","Heavily_Templated_Barman-100", "Heavily_Templated_Logistics-100","Moderately_Templated_Logistics-100", "Natural_Logistics-100"])
Parser.add_argument("--index_start", default="1", help="index to start generating result from (inclusive)")
Parser.add_argument("--index_end", default="51", help="index to end generating result from (exclusive)")
Parser.add_argument("--prediction_type", default="llm-as-formalizer", help="llm-as-formalizer or llm-as-planner",choices=["llm-as-formalizer", "llm-as-planner"])
Parser.add_argument("--prompt_version", default="pddl_instruction", help="rag or not",choices=["pddl_instruction", "pddl_instruction_rag"])
Parser.add_argument("--csv_result", default="True", help="get full output as csv file", action='store_true')
Parser.add_argument("--type", default="rag_refine", choices=["rag", "formalize", "rag_refine, steady"])



def standardize_blocks(plan):
    new_plan = re.sub(
        r'\b(?:b|block)?\s*(\d+)\b',
        r' block\1',
        plan,
        flags=re.IGNORECASE
    )
    return new_plan


def plan_to_path(domain, plan, plan_filepath):
    if domain not in ["barman", "logistics"]:
        plan = plan.strip().lower().replace('-', '').replace('_', '')
    else:
        plan = plan.strip().lower()
    if domain == "blocksworld":
        plan = standardize_blocks(plan)
    with open(plan_filepath, 'w') as plan_file:
        plan_file.write(plan)

    return plan, plan_filepath


def validate_plan(domain, problem_file_path, plan_filepath):
    validate_executable = "../../../bin/Validate"
    if domain == "blocksworld":
        domain_path = '../../data/textual_blocksworld/BlocksWorld-100_PDDL/domain.pddl'
    elif domain == "mystery_blocksworld":
        domain_path = '../../data/textual_mystery_blocksworld/Mystery_BlocksWorld-100_PDDL/domain.pddl'
    elif domain == "barman":
        domain_path = '../../data/textual_barman/Barman-100_PDDL/domain.pddl'
    elif domain == "logistics":
        domain_path = '../../data/textual_logistics/Logistics-100_PDDL/domain.pddl'

    command = [validate_executable, "-v", domain_path, problem_file_path, plan_filepath]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"Validation Output:\n{result.stdout}"

    except subprocess.CalledProcessError as e:
        return f"Error:\n{e.output}"


def validate_plan_batch(domain, data, model, index_start, index_end, prediction_type, csv_result, data_type):
    if '/' in model:
        _, model_name = model.split('/')
    else:
        model_name = model

    problem_names = []
    plan_found = []
    pddl_errors = []
    plans = []
    val_results = []
    is_plan_correct = []
    solvability = 0
    correctness = 0
    plan_not_found_error = 0
    syntax_error = 0

    for problem_number in range(index_start, index_end):
        problem_name = "p0" + str(problem_number) if problem_number < 10 else "p" + str(problem_number)
        print(f"Running {problem_name}")
        problem_names.append(problem_name)
        if prediction_type == "llm-as-formalizer":
            # plan_file = f"../../../deepseek-R1_results/Heavily_Templated/{problem_name}/{problem_name}_{model_name}_plan.txt"
            plan_file = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_plan.txt'
        else:
            # plan_file = f'../../../deepseek-R1_results/Heavily_Templated/{problem_name}/{problem_name}_{model_name}_plan.txt'
            plan_file = f'../../output/llm-as-planner/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_plan.txt'
        print(os.path.exists(plan_file))
        print(plan_file)
        if os.path.exists(plan_file):


            plan_found.append("yes")
            pddl_errors.append('')
            solvability += 1
            if domain == "blocksworld":
                problem_file = f'../../data/textual_blocksworld/BlocksWorld-100_PDDL/{problem_name}.pddl'
            elif domain == "mystery_blocksworld":
                problem_file = f'../../data/textual_mystery_blocksworld/Mystery_BlocksWorld-100_PDDL/{problem_name}.pddl'
            elif domain == "barman":
                problem_file = f'../../data/textual_barman/Barman-100_PDDL/{problem_name}.pddl'
            elif domain == "logistics":
                problem_file = f'../../data/textual_logistics/Logistics-100_PDDL/{problem_name}.pddl'

            plan = open(plan_file).read()

            if prediction_type == "llm-as-formalizer":
                new_plan_file = f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_plan_VAL.txt'
            else:
                new_plan_file = f'../../output/llm-as-planner/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_plan_VAL.txt'

            standard_plan, _ = plan_to_path(domain, plan, new_plan_file)
            plans.append(standard_plan)
            val_result = validate_plan(domain, problem_file, new_plan_file)
            val_results.append(val_result)

            if "Error" in val_result or "Failed" in val_result:
                is_plan_correct.append("no")
                print(f"corrcetness error:{problem_name}")
            else:
                is_plan_correct.append("yes")
                correctness += 1
        else:
            if os.path.exists(f"../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_0.txt"):
                error = open(
                    f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_0.txt').read()
            elif os.path.exists(f"../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_1.txt"):
                error = open(
                    f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_1.txt').read()
            elif os.path.exists(f"../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_2.txt"):
                error = open(
                    f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_2.txt').read()
            elif os.path.exists(f"../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_3.txt"):
                error = open(
                    f'../../output/llm-as-formalizer/{domain}/{data}/{model_name}/{data_type}/{problem_name}_pddl_instruction/{problem_name}_{model_name}_error_3.txt').read()

            if "OK" in error or "ff" in error:
                plan_not_found_error += 1
            else:
                syntax_error += 1

            plan_found.append("no")
            pddl_errors.append(error)
            plans.append('')
            val_results.append('')
            is_plan_correct.append('')
            print(f"Solvability error:{problem_name}")

    if csv_result:
        all_results_dict = {"problem_number": problem_names, "plan_found": plan_found,
                            "error, if not found": pddl_errors, "plan, if found": plans, "val_result": val_results,
                            "is_plan_correct": is_plan_correct}
        all_results = pd.DataFrame(all_results_dict)
        result_path = f'../../output/{prediction_type}/{domain}/{data}/{model_name}/{prediction_type}_{domain}_{data}_{model_name}_results.csv'
        all_results.to_csv(result_path)

    print(f"Solvability: {solvability if prediction_type == 'llm-as-formalizer' else '---'}/{index_end - index_start}")
    print(f"Correctness: {correctness}/{index_end - index_start}")
    print(f"Plan_not_found_error: {plan_not_found_error}/{index_end - index_start}")
    print(f"Syntax_error: {syntax_error}/{index_end - index_start}")

if __name__ == "__main__":
    args = Parser.parse_args()
    DOMAIN = args.domain
    MODEL = args.model
    DATA = args.data
    INDEX_START = eval(args.index_start)
    INDEX_END = eval(args.index_end)
    PREDICTION_TYPE = args.prediction_type
    CSV_RESULT = args.csv_result
    TYPE = args.type

    validate_plan_batch(domain=DOMAIN, data=DATA, model=MODEL, index_start=INDEX_START, index_end=INDEX_END,
                        prediction_type=PREDICTION_TYPE, csv_result=CSV_RESULT,data_type=TYPE)