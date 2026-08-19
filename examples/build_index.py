import json
from voiceraghh.pipeline import VoiceRAG


with open("sample_data.json") as f:
    data = json.load(f)

texts = []
for item in data:
    eng_passages = item.get("passages", {}).get("English_passages", [])
    for p in eng_passages:
        texts.append(p)
    
    if item.get("Eng_Query"):
        texts.append(item["Eng_Query"])
    if item.get("Eng_Answer"):
        texts.append(item["Eng_Answer"])

texts = list(set(texts))
print(f"Indexing {len(texts)} unique texts...")

rag = VoiceRAG()
rag.build_index(texts)
rag.vectorstore.save("./data/index")
print("Index saved to ./data/index")
