sweagent run \
  --agent.model.name=grpo-Qwen/Qwen2.5-3B-Instruct \
  --agent.model.max_input_tokens=10240 \
  --agent.model.grpo_lora_rank=64 \
  --agent.model.log_to_wandb=False \
  --agent.model.wandb_run_name=grpo-test \
  --agent.model.outputs_folder=/home/stevenabreu/grpo-implement/swea-outputs \
  --agent.model.num_generations=8 \
  --config=config/default_no_fcalls_shortest.yaml \
  --agent.model.per_instance_cost_limit=1.00 \
  --env.repo.github_url=https://github.com/SWE-agent/test-repo \
  --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1

#   --agent.model.max_input_tokens=128000 \
#   --config=config/default_no_fcalls_short.yaml \
#   --config=config/default_no_fcalls.yaml \
#   --agent.tools.parse_function=thought_action \
