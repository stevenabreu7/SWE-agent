from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
import wandb


def load_model_tokenizer(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
):
    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    return model, tokenizer


def load_trainer(
    model,
    tokenizer,
    num_generations: int,
    outputs_folder: str,
    run_name: str,
    log_to_wandb: bool = False,
    max_prompt_length: int = 256,
    max_completion_length: int = 200,
    max_steps: int = 250,
    reward_funcs: list = None,
    dataset: Dataset = None,
):
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "constant",  # NOTE: issue with step reset..
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = num_generations, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = max_steps,
        save_steps = 1,
        max_grad_norm = 0.1,
        report_to = "wandb" if log_to_wandb else "none", # Change from "none" to "wandb"
        run_name = run_name, # Set your desired run name
        output_dir = outputs_folder,
    )

    reward_funcs = [
        lambda completions, **kwargs: [0.0] * len(completions)
    ] if reward_funcs is None else reward_funcs
    dataset = get_gsm8k_questions() if dataset is None else dataset
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )
    return trainer


def get_gsm8k_questions(split = "train", idx_offset = 0, num_examples = 1) -> Dataset:
    split = split if num_examples is None else f"{split}[{idx_offset}:{idx_offset+num_examples}]"
    data = load_dataset('openai/gsm8k', 'main', split=split) # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': "You are a helpful assistant that can answer questions."},
            {'role': 'user', 'content': x['question']}
        ],
    }) # type: ignore
    return data # type: ignore


def init_wandb(
    run_name: str,
    training_args: GRPOConfig,
    log_to_wandb: bool = False,
    entity: str = "rug-minds",
    project: str = "GRPO-LLM",
):
    if log_to_wandb:
        wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            config=vars(training_args)
        )
