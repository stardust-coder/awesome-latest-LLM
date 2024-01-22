# Awesome latest LLMs

Keeping up with the latest LLMs !

**NEWS**

* 2024.1 Phi-2のMoE, Phixtralが公開されました！


<details>

<summary>History</summary>

- 2023.12 Phi-2のライセンスがMITに変更されました！  
- 2023.12 ELYZAから日本語[13Bモデル](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b)がリリースされました.  
- 2023.12 東工大から[Swallow](https://tokyotech-llm.github.io)がリリースされました.  
- 2023.12 MistralAIから[Mixtral-8x7B](https://github.com/open-compass/MixtralKit)がリリースされました.    
- 2023.12 [日本語LLMの学習データを問題視する記事](https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis)が公開されました.

</details>

## English-centric

|When? | Name |  HF?  | Size | License | pretraining/base | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.1| [Self-Rewarding(Meta)]() | [arxiv](https://arxiv.org/pdf/2401.10020.pdf) | 70B | Llama2 | Llama2| - | DPO |
<!-- |2024.1| [LLaMa-Pro-8B(Tencent)]() | [HF](https://huggingface.co/TencentARC/LLaMA-Pro-8B) | 8B | Llama2 ||| -->
|2024.1| [Phixtral]() | [HF](https://huggingface.co/mlabonne/phixtral-4x2_8) | 2.7Bx4 | MIT |||MoE|
|2023.12| [LongNet(Microsoft)](https://github.com/microsoft/torchscale) | [arXiv](https://arxiv.org/pdf/2307.02486.pdf) | - | apache-2.0 | [MAGNETO](https://arxiv.org/pdf/2210.06423.pdf)| input 1B token| |
|2023.12| [Phi-2(Microsoft)]() | [HF](https://huggingface.co/microsoft/phi-2) | 2.7B | MIT |||
|2023.12| [gigaGPT(Cerebras)](https://github.com/Cerebras/gigaGPT) | | 70B, 175B | apache-2.0 | | |
<!-- |2023.12| [Amber](https://www.llm360.ai) | [HF](https://huggingface.co/LLM360/Amber) | 7B | apache-2.0 | Llama|| totally open| -->
|2023.12| [Mixtral-8x7B](https://github.com/open-compass/MixtralKit)| [HF](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 8x7B | apache-2.0 |||MoE, [offloading](https://github.com/dvmazur/mixtral-offloading)|
|2023.12| [Mamba](https://github.com/state-spaces/mamba)| [HF](https://huggingface.co/state-spaces/mamba-2.8b) | 2.8B | apache-2.0 | based on state space model| | 
<!-- |2023.11| [Orca2(Microsoft)]() | [HF](https://huggingface.co/microsoft/Orca-2-13b) | 13B | MSRA-license| based on Llama2||| -->
|2023.11| [QWen(Alibaba)](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen/Qwen-72B) | 72B | [license](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)| 3T tokens | |beats Llama2|
|2023.9| [TinyLlama](https://github.com/jzhang38/TinyLlama) | [HF](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | apache-2.0 | 1.1B | based on Llama, 3T token |  | |
|2023.9| [XWin](https://github.com/Xwin-LM/Xwin-LM) | [HF](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1)  | 70B | Llama2 |based on Llama2| also codes and math|
|2023.7| [Llama2(Meta)](https://ai.meta.com/llama/) | [HF](https://huggingface.co/meta-llama) | 70B | Llama2 |based on Llama2| chat-hf seems the best|
<!-- |2023.9| [Phi-1.5(Microsoft)](https://arxiv.org/abs/2309.05463) | [HF](https://huggingface.co/microsoft/phi-1_5) | 1.3B| MSRA-license||textbooks| -->
|| [name]() | [HF]() | | |||


- PaLM(540B), GPT-4 are not open.
- MoE:mixture of experts

## Japanese-centric

|When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2023.12| [ELYZA-japanese-Llama-2-13b](https://note.com/elyza/n/n5d42686b60b7) | [HF](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b) | 13B | | Llama-2-13b-chatベース |
|2023.12| [Swallow(東工大)](https://tokyotech-llm.github.io) | [HF](https://huggingface.co/tokyotech-llm) | 70B | | Llama2-70Bベース |
|2023.11| [StableLM(StabilityAI)](https://ja.stability.ai/blog/japanese-stable-lm-beta) | [HF](https://huggingface.co/stabilityai/japanese-stablelm-base-beta-70b) | 70B | | Llama2-70Bベース |
|2023.10| [LLM-jp]() | [HF](https://huggingface.co/llm-jp) | 13B | |
|| [name]() | [HF]() | | |

See more on [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm)

## Medical　Adaptation including Multi-modality

|When? | Name |  HF?  | Size | License | pretraining | finetuning/continual | test | misc.|
|---|---|---|---|---|---|---|---|---|
|2024.1| [AMIE(Google)](https://arxiv.org/pdf/2401.05654.pdf) | not open | - | - | based on PaLM 2 |  |  | EHR| 
|2023.12| [Medprompt(Microsoft)]() | not open | - | - | GPT-4 | none |  |multi-modal| 
|2023.11| [Meditron(EPFL)](https://github.com/epfLLM/meditron) | [HF](https://huggingface.co/epfl-llm/meditron-70B) | 70B | Llama2 | Llama2, 48.1B | | 4 Q&As | |
|2023.8| [BioMedGPT(Luo et al.)](https://github.com/PharMolix/OpenBioMed) | [HF]() | 10B | |
|2023.8| [PMC-LLaMa](https://github.com/chaoyi-wu/PMC-LLaMA)| [HF]() | 13B | |
|2023.7| [Med-Flamingo](https://github.com/snap-stanford/med-flamingo) | [HF]() | 8.3B| ? | OpenFlamingo | MTB | Visual USMLE|based on Flamingo |
|2023.7| [LLaVa-Med(Microsoft)](https://github.com/microsoft/LLaVA-Med) | [HF](https://huggingface.co/microsoft/llava-med-7b-delta) | 13B | - | LLaVa| medical dataset |VAQ-RAD, SLAKE, PathVQA|multi-modal| 
|2023| [Med-PaLM M(Google)](https://arxiv.org/abs/2307.14334) | not open | | - | PaLM2 | | |multi-modal| 
|2023.5| [Med-PaLM2(Google)](https://arxiv.org/abs/2305.09617) | not open |340B | - | PaLM2 | | |
|2022.12| [Med-PaLM(Google)](https://arxiv.org/abs/2212.13138) | not open | 540B| - | PaLM | | | |
|| [name]() | [HF]() | | |


### Lists of dataset (medical)

Only Text
- [MedQA](https://github.com/jind11/MedQA) （USMLE）
- [MedMCQA](https://arxiv.org/abs/2203.14371)
- [PubMedQA](https://arxiv.org/abs/1909.06146)
- MMLU-Medical
- IgakuQA（Japanese National Medical License Exam）


Image + Text
- MTB: chopped cleaned text and images collected from 4721 textbooks.
- PMC-15M : the largest biomedical image-text dataset
- PMC-OA : 1.6M image-caption pairs
- [MedICaT](https://github.com/allenai/medicat): image, caption, textual reference
- [VQA-RAD](https://osf.io/89kps/) : 3515 question–answer pairs on 315 radiology images.
- SLAKE : bilingual dataset (English&Chinese) consisting of 642 images and 14,028 question-answer pairs
- PathVQA : 
- Visual USMLE : 618 USMLE-style QA
- [MedVTE](https://github.com/ynklab/MedVTE): numeric understanding
- [MedAlign(Stanford)](https://github.com/som-shahlab/medalign)
- MIMIC-IV : ECG-caption dataset
- [ECG-QA](https://github.com/Jwoo5/ecg-qa)


### Curations
- [Clinical NLP 2023](https://clinical-nlp.github.io/2023/resources.html)

See more on [He et al.(2023)](https://arxiv.org/pdf/2310.05694.pdf).