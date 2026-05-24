from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
import os
from pdb import set_trace
# from inference_serial_test_set import get_prompt
from inference_utils import ModelArguments, extract_audio_features, get_audio_path_list, get_fbank, get_prompt, maybe_init_qwen3_embedding_model, prepare_model_inputs

model_args = ModelArguments(
    model_name_or_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs192_step40000_2s/checkpoint-30000",
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B")
tokenizer.padding_side = "left"
model = SALMONN.from_pretrained(
    model_args.model_name_or_path,
    config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json")),
    model_args=model_args,
    torch_dtype="auto",
    device_map=0
)
maybe_init_qwen3_embedding_model(model, model_args)
fbank = get_fbank(model_args)
def makeInference(data):
    texts = []
    user_prompts = []
    audio_paths = []

    for sample in data:
        audio_path_list = get_audio_path_list(sample)
        audio_num = len(audio_path_list)
        audio_paths += audio_path_list
        if "expand_wav" in sample.keys():
            audio_num += len(sample["expand_wav"])
            audio_paths += sample["expand_wav"] # for some reason expand_wav is a list not a string!
        prompt = get_prompt(sample)
        messages = [
            {"role": "user", "content": "<audio>"*audio_num+prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
        user_prompts.append(prompt)

    feature, feature_lens, raw_wavs = extract_audio_features(audio_paths, fbank, model, model_args)

    model_inputs = prepare_model_inputs(texts, tokenizer, model)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        fbank_feature=feature,
        fbank_feature_len=feature_lens,
        raw_wavs=raw_wavs,
        user_prompts=user_prompts,
        max_new_tokens=5000
    )

    for generated_id in generated_ids:
        output_ids = generated_id.tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        print("="*60)
        if len(data) == 1:
            print("Single Inference Result")
        else:
            print("Batch Inference Results")
        print("="*60)

        print("="*60)
        print("Start of Cleaned Content")
        print("="*60)
        cleaned_content = content.split("</think>")[-1].strip()
        print(cleaned_content)


def main():
    contents = [
        {
            "path": "/mnt/bn/audio-visual-llm-data/chenxianzhao/dataset/wiki_QA/wiki_qa_test_for_tts_v2/audios/test_Q992.wav",
            "text": "Frank Lloyd Wright (born Frank Lincoln Wright, June 8, 1867 – April 9, 1959) was an American architect, interior designer, writer and educator, who designed more than 1,000 structures and completed 532 works.",
            "task": "speech_query",
            "question": "What building in NYC did Frank LLoyd Wright Design",
            "answer": [
                "Frank Lloyd Wright (born Frank Lincoln Wright, June 8, 1867 – April 9, 1959) was an American architect, interior designer, writer and educator, who designed more than 1,000 structures and completed 532 works.",
                "Wright believed in designing structures which were in harmony with humanity and its environment, a philosophy he called organic architecture .",
                "This philosophy was best exemplified by his design for Fallingwater (1935), which has been called \"the best all-time work of American architecture\".",
                "Wright was a leader of the Prairie School movement of architecture and developed the concept of the Usonian home, his unique vision for urban planning in the United States.",
                "His work includes original and innovative examples of many different building types, including offices, churches, schools, skyscrapers, hotels, and museums.",
                "Wright also designed many of the interior elements of his buildings, such as the furniture and stained glass .",
                "Wright authored 20 books and many articles and was a popular lecturer in the United States and in Europe.",
                "His colorful personal life often made headlines, most notably for the 1914 fire and murders at his Taliesin studio .",
                "Already well known during his lifetime, Wright was recognized in 1991 by the American Institute of Architects as \"the greatest American architect of all time.\""
            ]
        },
        {
            "path": "/mnt/bn/audio-visual-llm-data/chenxianzhao/dataset/wiki_QA/wiki_qa_test_for_tts_v2/audios/test_Q995.wav",
            "text": "A medieval ship flag captured by forces from Lübeck in the 1420s showed the arms of Denmark, Sweden, Norway and Pomerania.",
            "task": "speech_query",
            "question": "what does it mean for  a ship to be flagged by another country",
            "answer": [
                "A medieval ship flag captured by forces from Lübeck in the 1420s showed the arms of Denmark, Sweden, Norway and Pomerania.",
                "The original flag was destroyed during a World War II attack on the city, but a 19th century copy remains in Frederiksborg Palace , Denmark.",
                "The saint accompanying the Virgin Mary and infant Christ is Saint James the Greater , identified by his scallop shell emblem.",
                "A maritime flag is a flag designated for use on ships , boats , and other watercraft.",
                "Naval flags are considered important at sea and the rules and regulations for the flying of flags are strictly enforced.",
                "The flag flown is related to the country of registration : so much so that the word \"flag\" is often used symbolically as a synonym for \"country of registration\"."
            ]
        },
        {
            "path": "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/test/wav/id10270/x6uYqmx31kE/00001.wav",
            "text": "Yes",
            "task": "speaker_verification",
            "expand_wav": [
                "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/test/wav/id10270/8jEAjG6SegY/00008.wav"
            ]
        },
        {
            "path": "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/test/wav/id10270/x6uYqmx31kE/00001.wav",
            "text": "No",
            "task": "speaker_verification",
            "expand_wav": [
                "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/test/wav/id10300/ize_eiCFEg0/00003.wav"
            ]
        },
         {
            "path": "/mnt/bn/audio-visual-llm-data/datasets/GigaSpeech/preprocessed_data/subset_test/audio/POD1000000005/POD1000000005_S0000000.wav",
            "text": "Hey friends i don't know about you but when i was a kid i knew absolutely nothing about money",
            "task": "asr"
        },
        {
            "path": "/mnt/bn/audio-visual-llm-data/datasets/GigaSpeech/preprocessed_data/subset_test/audio/POD1000000005/POD1000000005_S0000001.wav",
            "text": "Well there's a new show from marketplace and brains on that will help address just that",
            "task": "asr"
        },
    ]
    makeInference(contents)

    for content in contents:
        makeInference([content])


if __name__ == "__main__":
    main()