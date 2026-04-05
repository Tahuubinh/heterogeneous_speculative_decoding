import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

model_name = "state-spaces/mamba-130m"

# tokenizer vẫn dùng HF (OK)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# load model đúng cách
model = MambaLMHeadModel.from_pretrained(model_name).to("cuda")

model.eval()

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# forward
# with torch.no_grad():
#     outputs = model.generate(
#         input_ids=inputs["input_ids"],
#         max_length=50,
#     )

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=50,
        cg=True, # Bật CUDA Graph giúp tăng tốc độ sinh text đáng kể
        # eos_token_id=tokenizer.eos_token_id # (Tùy chọn) Dừng lại khi gặp token kết thúc
    )

print(tokenizer.decode(outputs[0]))