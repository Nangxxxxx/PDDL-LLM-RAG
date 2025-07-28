#python llm_as_formalizer.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 57 --index_end 101
#python llm_as_formalizer_example.py  --domain blocksworld --model llama-4-maverick-17b-128e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 4 --index_end 51
#python llm_as_formalizer_continue_doc.py  --domain blocksworld --model qwen3-8b --data Heavily_Templated_BlocksWorld-100 --index_start 46 --index_end 51
#python llm_as_formalizer_continue.py  --domain blocksworld --model qwq-32b --data Heavily_Templated_BlocksWorld-100 --index_start 19 --index_end 51
#python llm_as_formalizer_steady.py  --domain blocksworld --model qwen3-8b --data Heavily_Templated_BlocksWorld-100 --index_start 8 --index_end 51
#python llm_as_formalizer_continue.py  --domain barman --model deepseek-reasoner --data Heavily_Templated_Barman-100 --index_start 42 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain barman --model deepseek-reasoner --data Heavily_Templated_Barman-100 --index_start 13 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain blocksworld --model qwq-32b --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 51 --index_end 101 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain barman --model qwq-32b --data Heavily_Templated_Barman-100 --index_start 98 --index_end 101 --solver dual-bfws-ffparser

#
#python run_solver.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 23 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain barman --model qwq-32b --data Heavily_Templated_Barman-100 --index_start 7 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain barman --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Barman-100 --index_start 23 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain barman --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_Barman-100 --index_start 14 --index_end 51 --solver dual-bfws-ffparser --type steady

#python run_solver.py  --domain blocksworld --model qwen3-8b --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver.py  --domain blocksworld --model qwq-32b --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver.py  --domain blocksworld --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver.py  --domain blocksworld --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser

#python run_solver.py  --domain logistics --model qwen3-8b --data Heavily_Templated_Logistics-100 --index_start 23 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain logistics --model qwq-32b --data Heavily_Templated_Logistics-100 --index_start 18 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain logistics --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Logistics-100 --index_start 25 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain logistics --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_Logistics-100 --index_start 27 --index_end 51 --solver dual-bfws-ffparser --type steady

#python run_solver.py  --domain mystery_blocksworld --model qwen3-8b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 26 --index_end 51 --solver dual-bfws-ffparser --type steady
#python run_solver.py  --domain mystery_blocksworld --model qwq-32b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 21 --index_end 51 --solver dual-bfws-ffparser --type steady
python run_solver.py  --domain mystery_blocksworld --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 12 --index_end 13 --solver dual-bfws-ffparser --type steady
python run_solver.py  --domain mystery_blocksworld --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 31 --index_end 51 --solver dual-bfws-ffparser --type steady


#python run_solver_error_rag.py  --domain logistics --model qwen3-8b --data Heavily_Templated_Logistics-100 --index_start 26 --index_end 51 --solver dual-bfws-ffparser


#python run_solver_error_rag.py  --domain mystery_blocksworld --model qwq-32b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain mystery_blocksworld --model qwen3-8b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 9 --index_end 51 --solver dual-bfws-ffparser

#python run_solver_error_rag.py  --domain mystery_blocksworld --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 56 --index_end 101 --solver dual-bfws-ffparser
#python llm_as_formalizer_separate.py  --domain mystery_blocksworld --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 44 --index_end 51
#python llm_as_formalizer_separate.py  --domain mystery_blocksworld --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 28 --index_end 51




#python llm_as_formalizer.py  --domain logistics --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Logistics-100 --index_start 11 --index_end 51
#python llm_as_formalizer_separate.py  --domain logistics --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Logistics-100 --index_start 15 --index_end 51
#python run_solver_refine.py  --domain logistics --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain logistics --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser

#python llm_as_formalizer.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 17 --index_end 51
#python llm_as_formalizer_separate.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 10 --index_end 51
#python run_solver_refine.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-llama-70b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 9 --index_end 51 --solver dual-bfws-ffparser

##python llm_as_formalizer.py  --domain logistics --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Logistics-100 --index_start 14 --index_end 51
#python llm_as_formalizer_separate.py  --domain logistics --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 51
#python run_solver_refine.py  --domain logistics --model deepseek/deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain logistics --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#
##python llm_as_formalizer.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 11 --index_end 51
#python llm_as_formalizer_separate.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 2 --index_end 51
#python run_solver_refine.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain mystery_blocksworld --model deepseek/deepseek-r1-distill-qwen-14b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser


#python run_solver_refine.py  --domain mystery_blocksworld --model qwen3-8b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 83 --index_end 101 --solver dual-bfws-ffparser


#python run_solver_error_rag.py  --domain logistics --model qwq-32b --data Heavily_Templated_Logistics-100 --index_start 5 --index_end 101 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain logistics --model qwen3-8b --data Heavily_Templated_Logistics-100 --index_start 1 --index_end 101 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain mystery_blocksworld --model qwq-32b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 101 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain mystery_blocksworld --model qwen3-8b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 101 --solver dual-bfws-ffparser

#python run_solver_refine.py  --domain mystery_blocksworld --model qwq-32b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain mystery_blocksworld --model qwq-32b --data Heavily_Templated_Mystery_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_error_rag.py  --domain blocksworld --model deepseek-r1-distill-llama-70b --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 51 --solver dual-bfws-ffparser
#python llm_as_formalizer_separate.py  --domain barman --model qwq-32b --data Heavily_Templated_Barman-100 --index_start 51 --index_end 101
#python llm_as_formalizer_separate.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 82 --index_end 101
#python llm_as_formalizer_separate.py  --domain barman --model meta-llama/llama-4-maverick-17b-128e-instruct --data Heavily_Templated_Barman-100 --index_start 1 --index_end 101
#python llm_as_formalizer_separate.py  --domain barman --model meta-llama/llama-4-scout-17b-16e-instruct --data Heavily_Templated_Barman-100 --index_start 1 --index_end 101
#python run_solver_error_rag.py  --domain barman --model qwq-32b --data Heavily_Templated_Barman-100 --index_start 39 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain barman --model qwq-32b --data Heavily_Templated_Barman-100 --index_start 22 --index_end 51 --solver dual-bfws-ffparser
#python llm_as_formalizer_separate.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 32 --index_end 51
#python run_solver_error_rag.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 36 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain barman --model qwen3-8b --data Heavily_Templated_Barman-100 --index_start 14 --index_end 51 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain blocksworld --model llama-4-scout-17b-16e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 26 --index_end 33 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain blocksworld --model llama-4-scout-17b-16e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 34 --index_end 36 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain blocksworld --model llama-4-scout-17b-16e-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 40 --index_end 44 --solver dual-bfws-ffparser
#python run_solver_rag_re_generate.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_rag_refine.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_steady_refine.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_rewrite_rag.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Moderately_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Moderately_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_todo_refine.py  --domain blocksworld --model qwen2.5-coder-7b-instruct --data Heavily_Templated_BlocksWorld-100 --index_start 1 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_rag_refine.py  --domain blocksworld --model deepseek-reasoner --data Moderately_Templated_BlocksWorld-100 --index_start 85 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_refine.py  --domain blocksworld --model deepseek-reasoner --data Moderately_Templated_BlocksWorld-100 --index_start 97 --index_end 100 --solver dual-bfws-ffparser
#python run_solver_steady_refine.py  --domain blocksworld --model deepseek-reasoner --data Moderately_Templated_BlocksWorld-100 --index_start 82 --index_end 100 --solver dual-bfws-ffparser
