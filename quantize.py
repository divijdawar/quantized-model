from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import NVFP4

model_id = "cerebras/DeepSeek-V3.2-REAP-345B-A37B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

DATASET_ID = "GAIR/LIMI"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    text = ""
    for msg in example["conversations"]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n\n"
    return {"text": text}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = [
    SpinQuantModifier(rotations=["R3"]),
    QuantizationModifier(
        config_groups={
            "all_linears": QuantizationScheme(
                targets=["Linear"],
                input_activations=NVFP4["input_activations"],
            ),
        }
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-r3-attention-nvfp4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
