import json

args = {
    "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
    "do_train": True,
    "dataset": "mllm_demo,identity",
    "template": "qwen2_vl",
    "finetuning_type": "lora",
    "lora_target": "all",
    "output_dir": "qwen2_vl_lora",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
}


with open("qwen2_vl_lora.json", "w", encoding="utf-8") as f:
    json.dump(args, f, ensure_ascii=False, indent=4)
