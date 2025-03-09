## run locally in ollama
sweagent run \
  --agent.model.name=ollama/qwen2.5:3b-instruct \
  --agent.model.max_input_tokens=128000 \
  --config=config/default_no_fcalls.yaml \
  --agent.model.per_instance_cost_limit=1.00 \
  --env.repo.github_url=https://github.com/SWE-agent/test-repo \
  --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1
