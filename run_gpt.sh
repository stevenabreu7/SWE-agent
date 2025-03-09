sweagent run \
  --agent.model.name=gpt-4o \
  --config=config/default_no_fcalls.yaml \
  --agent.model.per_instance_cost_limit=1.00 \
  --env.repo.github_url=https://github.com/SWE-agent/test-repo \
  --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1
