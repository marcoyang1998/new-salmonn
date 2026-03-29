import json
from pdb import set_trace

def get_jsonl_data(path):
    with open(path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        return data
            

def compare(single_data, batch_data):
    counter = 0
    for i, (sample1, sample2) in enumerate(zip(single_data, batch_data)):
        if sample1.get("response") != sample2.get("response"):
            counter += 1
            print(f"Difference found at sample {i + 1}:")
            print(f"  Single Inference File Response: {sample1.get('response')}")
            print(f"  Batch Inference File Response: {sample2.get('response')}")
    print(f"Total differences: {counter}")

def main():
    single_path = "/mnt/bn/audio-visual-llm-data6/terumi/Salmon_Inference_test/test.jsonl"
    batch_path = "/mnt/bn/audio-visual-llm-data6/terumi/Salmon_Inference_test/batch_test.jsonl"
    
    single_data = get_jsonl_data(single_path)
    batch_data = get_jsonl_data(batch_path)
    
    compare(single_data, batch_data)

if __name__ == "__main__":
    main()