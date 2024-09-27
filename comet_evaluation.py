from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)

# Data must be in the following format:
data = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        "ref": "Can it be delivered between 10 to 15 minutes?"
    }
]
# Call predict method:
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output)
print(model_output.scores) # sentence-level scores
print(model_output.system_score) # system-level score

# Not all COMET models return metadata with detected errors.
print(model_output.metadata.error_spans) # detected error spans
