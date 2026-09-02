from transformers import pipeline

# Load FLAN-T5 model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

prompt = "Explain artificial intelligence in simple words."

output = generator(
    prompt,
    max_new_tokens=50
)

print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(output[0]["generated_text"])