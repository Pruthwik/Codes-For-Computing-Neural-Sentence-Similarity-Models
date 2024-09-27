from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
# model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# with reference
model_path_with_reference = download_model("Unbabel/wmt22-comet-da")
# without reference
model_path_without_reference = download_model("Unbabel/wmt22-cometkiwi-da")

# Load the model checkpoint:
model_ref = load_from_checkpoint(model_path_with_reference)
model_no_ref = load_from_checkpoint(model_path_without_reference)

# Data must be in the following format:
data_with_ref = [
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
model_output_ref = model_ref.predict(data_with_ref, batch_size=8, gpus=1)
print(model_output_ref)
print(model_output_ref.scores) # sentence-level scores
print(model_output_ref.system_score) # system-level score

# Not all COMET models return metadata with detected errors.
# print(model_output.metadata.error_spans) # detected error spans
data_no_ref = [
    {
        "src": "10 到 15 分钟可以送到吗",
        "mt": "Can I receive my food in 10 to 15 minutes?",
        # "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "Pode ser entregue dentro de 10 a 15 minutos?",
        "mt": "Can you send it for 10 to 15 minutes?",
        # "ref": "Can it be delivered between 10 to 15 minutes?"
    },
    {
        "src": "ପଞ୍ଜାବ-ହରିୟାଣା ସମେତ ଅନେକ ରାଜ୍ୟର କୃଷକମାନେ ଜାତୀୟ ରାଜଧାନୀ ଦିଲ୍ଲୀର ସୀମାରେ କେନ୍ଦ୍ର ସରକାରଙ୍କ ନୂତନ କୃଷି ନିୟମକୁ ବିରୋଧ କରୁଛନ୍ତି।",
        "mt": "केंद्र सरकार के नए कृषि कानूनों को लेकर पंजाब-हरियाणा समेत कई राज्यों के किसान राष्ट्रीय राजधानी दिल्ली की सीमाओं पर धरना प्रदर्शन कर रहे हैं ।"
    }
]
# Call predict method:
model_output_no_ref = model_no_ref.predict(data_no_ref, batch_size=8, gpus=1)
print(model_output_no_ref)
print(model_output_no_ref.scores) # sentence-level scores
print(model_output_no_ref.system_score) # system-level score
