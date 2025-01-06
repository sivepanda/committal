import pygit2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = './tuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def extract_staged_diff(repo_path="."):
    repo = pygit2.Repository(repo_path)
    diff = repo.diff(repo.head.target, context_lines = 3)
    return diff.patch

def generate_commit_message(diff):
    inputs = tokenizer(diff, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
