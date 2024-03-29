# Awesome latest LLMs

Keeping up with the latest LLMs !

**NEWS**
- 2024.3 Cohereからプロダクション向けCommand-Rがリリース!研究用に重みも公開.
- 2024.3 ELYZAからLlama2の追加学習日本語モデルのデモがリリースされました！
- 2024.3 東工大からMixtralの追加学習日本語モデル[Swallow-MX](), [Swallow-MS]()がリリースされました！👏
<details>

<summary>History</summary>
- 2024.2 GoogleからGeminiで用いられているLLM [Gemma](https://blog.google/technology/developers/gemma-open-models/)をオープンにするとのお達しが出ました!
- 2024.2 Kotoba Technologyと東工大から[日本語Mamba 2.8B](https://huggingface.co/kotoba-tech/kotomamba-2.8B-v1.0)が公開されました!
- 2024.2 Alibabaの[QWen](https://qwenlm.github.io/blog/qwen1.5/)が1.5にアップグレードされました！！
- 2024.2 Reka AIから21BでGemini Pro, GPT-3.5超えと発表されました.
- 2024.2 LLM-jpのモデルが更新されました！v1.1
- 2024.2 カラクリから70B日本語LLMが公開されました！
- 2024.1 [リコー](https://www.nikkei.com/article/DGXZRSP667803_R30C24A1000000/)が13B日本語LLMを発表しました！
- 2024.1 Phi-2のMoE, Phixtralが公開されました！
- 2023.12 Phi-2のライセンスがMITに変更されました！  
- 2023.12 ELYZAから日本語[13Bモデル](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b)がリリースされました.  
- 2023.12 東工大から[Swallow](https://tokyotech-llm.github.io)がリリースされました.  
- 2023.12 MistralAIから[Mixtral-8x7B](https://github.com/open-compass/MixtralKit)がリリースされました.    
- 2023.12 [日本語LLMの学習データを問題視する記事](https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis)が公開されました.
</details>

## English-centric


|When? | Name |  HF?  | Size(max) | License | pretraining/base | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.3| [Command-R(Cohere)](https://txt.cohere.com/command-r/) |[HF](https://huggingface.co/CohereForAI/c4ai-command-r-v01) | 35B | non commercial | || RAG capability |
|2024.2| [Aya(Cohere)](https://cohere.com/research/aya?ref=txt.cohere.com) |[HF](https://huggingface.co/CohereForAI/aya-101) | 13B | apache-2.0 | || multilingual |
|2024.2| [Gemma(Google)](https://blog.google/technology/developers/gemma-open-models/) | | 8.5B | || |application open for reseachers |
|2024.2| [Miqu](https://twitter.com/arthurmensch/status/1752737462663684344?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1752737462663684344%7Ctwgr%5Ecd2e234e5fa688c1a14852aa90158cd4f59facb4%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fgigazine.net%2Fnews%2F20240201-hugging-face-miqu-mistral-model%2F) | [HF](https://huggingface.co/miqudev/miqu-1-70b/tree/main) | 70B | none ||| leaked from Mistral |
|2024.2| [Reka Flash](https://reka.ai/reka-flash-an-efficient-and-capable-multimodal-language-model/) |  | 21B | ||| not public|
|2024.1| [Self-Rewarding(Meta)]() | [arxiv](https://arxiv.org/pdf/2401.10020.pdf) | 70B | Llama2 | Llama2| - | DPO |
|2024.1| [Phixtral]() | [HF](https://huggingface.co/mlabonne/phixtral-4x2_8) | 2.7Bx4 | MIT |||MoE|
|2023.12| [LongNet(Microsoft)](https://github.com/microsoft/torchscale) | [arXiv](https://arxiv.org/pdf/2307.02486.pdf) | - | apache-2.0 | [MAGNETO](https://arxiv.org/pdf/2210.06423.pdf)| input 1B token| |
|2023.12| [Phi-2(Microsoft)]() | [HF](https://huggingface.co/microsoft/phi-2) | 2.7B | MIT |||
|2023.12| [gigaGPT(Cerebras)](https://github.com/Cerebras/gigaGPT) | | 70B, 175B | apache-2.0 | | |
|2023.12| [Mixtral-8x7B](https://github.com/open-compass/MixtralKit)| [HF](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 8x7B | apache-2.0 |||MoE, [offloading](https://github.com/dvmazur/mixtral-offloading)|
|2023.12| [Mamba](https://github.com/state-spaces/mamba)| [HF](https://huggingface.co/state-spaces/mamba-2.8b) | 2.8B | apache-2.0 | based on state space model| | 
|2023.11| [QWen(Alibaba)](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen/Qwen-72B) | 72B | [license](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)| 3T tokens | | beats Llama2 |
|2023.10| [Self-RAG](https://github.com/AkariAsai/self-rag) | [HF](https://huggingface.co/selfrag) | apache-2.0 | 13B |  |  | critic model |
|2023.9| [TinyLlama](https://github.com/jzhang38/TinyLlama) | [HF](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | apache-2.0 | 1.1B | based on Llama, 3T token |  | |
|2023.9| [Xwin-LM](https://github.com/Xwin-LM/Xwin-LM) | [HF](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1)  | 70B | Llama2 |based on Llama2| also codes and math|
|2023.7| [Llama2(Meta)](https://ai.meta.com/llama/) | [HF](https://huggingface.co/meta-llama) | 70B | Llama2 | 2T tokens| chat-hf seems the best|
|| [name]() | [HF]() | | |||

<!-- |2024.1| [LLaMa-Pro-8B(Tencent)]() | [HF](https://huggingface.co/TencentARC/LLaMA-Pro-8B) | 8B | Llama2 ||| -->
<!-- |2023.12| [Amber](https://www.llm360.ai) | [HF](https://huggingface.co/LLM360/Amber) | 7B | apache-2.0 | Llama|| totally open| -->
<!-- |2023.11| [Orca2(Microsoft)]() | [HF](https://huggingface.co/microsoft/Orca-2-13b) | 13B | MSRA-license| based on Llama2||| -->
<!-- |2023.9| [Phi-1.5(Microsoft)](https://arxiv.org/abs/2309.05463) | [HF](https://huggingface.co/microsoft/phi-1_5) | 1.3B| MSRA-license||textbooks| -->
- PaLM(540B), PaLM2(340B) and GPT-4 are not open.
- MoE : mixture of experts

## Japanese-centric

|When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.3| [Swallow-MX(東工大)]() | [HF](https://huggingface.co/tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1) | 8x7B | | Mixtralベース |
|2024.2| [KARAKURI](https://karakuri.ai/seminar/news/karakuri-lm/) | [HF](https://huggingface.co/karakuri-ai/karakuri-lm-70b-v0.1) | 70B | cc-by-sa-4.0 | Llama2-70Bベース | | [note](https://note.com/ngc_shj/n/n46ced665b378?sub_rt=share_h)|
|2023.12| [ELYZA-japanese-Llama-2-13b](https://note.com/elyza/n/n5d42686b60b7) | [HF](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b) | 13B | | Llama-2-13b-chatベース |
|2023.12| [Swallow(東工大)](https://tokyotech-llm.github.io) | [HF](https://huggingface.co/tokyotech-llm) | 70B | | Llama2-70Bベース |
|2023.11| [StableLM(StabilityAI)](https://ja.stability.ai/blog/japanese-stable-lm-beta) | [HF](https://huggingface.co/stabilityai/japanese-stablelm-base-beta-70b) | 70B | | Llama2-70Bベース |
|2023.10| [LLM-jp](https://llm-jp.nii.ac.jp/blog/2024/02/09/v1.1-tuning.html) | [HF](https://huggingface.co/llm-jp) | 13B | DPO追加あり |
|| [name]() | [HF]() | | ||||

See more on [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm)


--- 

# Medical-Adaptation 

## Model
|When? | Name |  HF?  | Size | License | pretraining | finetuning/continual | test | misc.|
|---|---|---|---|---|---|---|---|---|
|2024.3| [AdaptLLM(Microsoft Research)](https://arxiv.org/pdf/2309.09530.pdf) | [HF](https://huggingface.co/AdaptLLM/medicine-LLM) | | | | | |  | ICLR2024 |
|2024.3| [Apollo](https://github.com/FreedomIntelligence/Apollo) | | | | | | |  | multilingual |
2024.2| [Health-LLM(Rutgersなど)](https://arxiv.org/pdf/2402.00746.pdf) | | | | | | | RAG |
|2024.2| [BioMistral](https://arxiv.org/pdf/2402.10373.pdf) | not open | 7B | - |  |  |  | | 
|2024.1| [AMIE(Google)](https://arxiv.org/pdf/2401.05654.pdf) | not open | - | - | based on PaLM 2 |  |  | EHR| 
|2023.12| [Medprompt(Microsoft)]() | not open | - | - | GPT-4 | none |  |multi-modal| 
|2023.12| [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | [HF](https://huggingface.co/AIgroup-CVM-utokyohospital/llama2-jmedlora-3000) | 70B | none | none | QLoRA | IgakuQA | Japanese, insufficient quality | 
|2023.11| [Meditron(EPFL)](https://github.com/epfLLM/meditron) | [HF](https://huggingface.co/epfl-llm/meditron-70B) | 70B | Llama2 | Llama2 | GAP-Replay(48.1B) | [dataset](img/meditron-testdata.png),[score](img/meditron-eval2.png) | |
|2023.8| [BioMedGPT(Luo et al.)](https://github.com/PharMolix/OpenBioMed) | [HF]() | 10B | |
|2023.8| [PMC-LLaMa](https://github.com/chaoyi-wu/PMC-LLaMA)| [HF]() | 13B | |
|2023.7| [Med-Flamingo](https://github.com/snap-stanford/med-flamingo) | [HF]() | 8.3B| ? | OpenFlamingo | MTB | Visual USMLE|based on Flamingo |
|2023.7| [LLaVa-Med(Microsoft)](https://github.com/microsoft/LLaVA-Med) | [HF](https://huggingface.co/microsoft/llava-med-7b-delta) | 13B | - | LLaVa| medical dataset | VAQ-RAD, SLAKE, PathVQA |multi-modal| 
|2023.7| [Med-PaLM M(Google)](https://arxiv.org/abs/2307.14334) | not open | | - | PaLM2 | | |multi-modal| 
|2023.5| [Almanac(Stanford)](https://arxiv.org/pdf/2303.01229.pdf), [journal](https://ai.nejm.org/doi/pdf/10.1056/AIoa2300068) | ? | ? | text-davinci-003 |  | | RAG |
|2023.5| [Med-PaLM2(Google)](https://arxiv.org/abs/2305.09617) | not open | 340B | - | PaLM2 | | |
|2022.12| [Med-PaLM(Google)](https://arxiv.org/abs/2212.13138) | not open | 540B| - | PaLM | | | |
|| [name]() | [HF]() | | |

See also [Awesome-Healthcare-Foundation-Models](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)

[医療ドメイン特化LLMの性能はどうやって評価する？](https://zenn.dev/hellorusk/articles/04a29974138c7b)

## Evaluation

| How? | Who? | example works | form of output | example datasets | 
|---|---|---|---|---|
| Medical Doctor | | [Med-Flamingo](), [Med-PaLM]() | | |
| BERT similarity score | [Zhang et al.]() | [Med-Flamingo]() | text generation | |
| Exact match (modulo puncutuation) | | [Med-Flamingo]() | short text generation | VQA-RAD | 
| Exact match | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | multiple choice question | IgakuQA | 
| Gestalt score | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | multiple choice question | IgakuQA | 
| Accuracy | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | [JMedLoRA(UTokyo)](https://arxiv.org/abs/2310.10083) | multiple choice question | IgakuQA | 



## Dataset

Only Text
- [MedQA](https://github.com/jind11/MedQA) （USMLE）
- [MedMCQA](https://arxiv.org/abs/2203.14371)
- [PubMedQA](https://arxiv.org/abs/1909.06146)
- MMLU-Medical : extracted from [MMLU](https://github.com/hendrycks/test)
- [PubHealth](https://github.com/neemakot/Health-Fact-Checking)
- [JMMLU](https://github.com/nlp-waseda/JMMLU) : Japanese-translated version of MMLU
- [IgakuQA（Japanese National Medical License Exam）](https://github.com/jungokasai/IgakuQA)
- [J-ResearchCorpus](https://huggingface.co/datasets/kunishou/J-ResearchCorpus)

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

Curations
- [Clinical NLP 2023](https://clinical-nlp.github.io/2023/resources.html)

See more on [He et al.(2023)](https://arxiv.org/pdf/2310.05694.pdf).