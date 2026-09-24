from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt ="What is artificial intelligence? Explain it in simple words for a beginner."
# Convert prompt into tokens
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

# Convert tokens back to text
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(generated_text)
import csv

with open("text_generation_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Prompt", "Generated Text"])
    writer.writerow([prompt, generated_text])

print("Result saved successfully to text_generation_results.csv")