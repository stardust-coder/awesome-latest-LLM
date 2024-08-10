# Awesome latest LLMs

Keeping up with the latest LLMs !

**NEWS**
- 2024.7 🔥東工大からLlama3の日本語継続学習モデルが発表！



<details>

<summary>History</summary>

- 2024.6 🔥ELYZAからLlama3の日本語継続学習モデルが発表！
- 2024.6 🔥Googleから27BのGemma2が公開！何が強みか教えて！
- 2024.6 🔥NVIDIAが340Bの巨大モデルを公開！publicにしては最大級
- 2024.6 🔥QWen2シリーズが登場！日本語も優秀！
- 2024.5 🔥MicrosoftからPhi-3シリーズが登場！
- 2024.5 🔥Stockmarkから100Bの日本語モデルがリリース!さすがGENIAC
- 2024.4 🔥MetaからLlama3がリリース!まずは8Bと70B!
- 2024.4 🔥CohereからCommand-R+がリリース!研究用に重みも公開.
- 2024.4 🔥Databricksより132BのMoEモデルが公開されました！大きい！
- 2024.3 Cohereからプロダクション向けCommand-Rがリリース!研究用に重みも公開.
- 2024.3 ELYZAからLlama2の追加学習日本語モデルのデモがリリースされました！
- 2024.3 東工大からMixtralの追加学習日本語モデル[Swallow-MX](), [Swallow-MS]()がリリースされました！👏
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

| When? | Name |  HF?  | Size(max) | License | pretraining/base | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.6| [Gemma2(Google)]() |[HF](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | 9B, 27B | gemma |  |  |  |
|2024.6| [Nemotron(NVIDIA)]() |[HF](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) | 340B |  | - | - |  |
|2024.6| [QWen2(Alibaba)]() |[HF](https://huggingface.co/Qwen/Qwen2-72B) | 7~72B | tongyi-qianwen | - | - |  |
|2024.4| [Phi-3(Microsoft)](https://arxiv.org/abs/2404.14219) |[HF](microsoft/Phi-3-medium-128k-instruct) | 3.8B, 13B | MIT |  Phi-3 datasets | - |  |
|2024.4| [Llama 3(Meta)](https://llama.meta.com/llama3/) |[HF](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | 70B | [META LLAMA3](https://llama.meta.com/llama3/license/) | || [extended to 120B](https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct) |
|2024.4| [Wizart-8x22B(Microsoft)]() |[HF](https://huggingface.co/microsoft/WizardLM-2-8x22B) | 8x22B | apache-2.0 | [Mixtral-8x22B(Mistral)](https://mistral.ai/news/mixtral-8x22b/) | | MoE, closed now |
|2024.4| [Mixtral-8x22B(Mistral)](https://mistral.ai/news/mixtral-8x22b/) |[HF](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1) | 8x22B | apache-2.0 | || MoE |
|2024.4| [Command-R+(Cohere)](https://txt.cohere.com/command-r/) |[HF](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | 104B | non commercial | || RAG capability |
|2024.4| [DBRX(Databricks)]() |[HF](https://huggingface.co/databricks/dbrx-instruct) | 132B | databricks | || MoE |
|2024.3| [Grok-1](https://github.com/xai-org/grok-1) | | 314B | | twitter | | MoE |
|2024.3| [BTX(Meta)](https://arxiv.org/pdf/2403.07816.pdf)|||||| MoE |
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

| When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.7| [LLama3-Swallow]() | [HF](https://huggingface.co/tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1) | 70B | Llama3 | Llama3  | |  |
|2024.6| [LLama3ELYZA-JP-8B](https://elyza.ai/news/2024/06/26/elyza-llm-for-jp%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA%E3%81%AE%E6%9C%80%E6%96%B0%E3%83%A2%E3%83%87%E3%83%ABllam) | [HF](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B) | 8B | Llama3 | Llama3  | | 70B not open |
|2024.6| [KARAKURI LM 8x7B](https://karakuri.ai/seminar/news/karakuri-lm-8x7b-instruct-v0-1/) | [HF](karakuri-ai/karakuri-lm-8x7b-chat-v0.1) | 8x7B | Apache-2.0 |  | | MoE |
|2024.5| [Stockmark-100B]() | [HF](stockmark/stockmark-100b) | 100B | MIT |  | | |
|2024.3| [youko(rinna)]() | [HF](https://huggingface.co/rinna/llama-3-youko-8b) | 8B | Llama3 | Llama3 | | |
|2024.3| [EvoLLM-JP]() | [HF](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) | 7B | MSR(non-commercial) | | | |
|2024.3| [Swallow-MX(東工大)]() | [HF](https://huggingface.co/tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1) | 8x7B | | Mixtralベース |
|2024.2| [KARAKURI 70B](https://karakuri.ai/seminar/news/karakuri-lm/) | [HF](https://huggingface.co/karakuri-ai/karakuri-lm-70b-v0.1) | 70B | cc-by-sa-4.0 | Llama2-70Bベース | | [note](https://note.com/ngc_shj/n/n46ced665b378?sub_rt=share_h)|
|2023.12| [ELYZA-japanese-Llama-2-13b](https://note.com/elyza/n/n5d42686b60b7) | [HF](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b) | 13B | | Llama-2-13b-chatベース |
|2023.12| [Swallow(東工大)](https://tokyotech-llm.github.io) | [HF](https://huggingface.co/tokyotech-llm) | 70B | | Llama2-70Bベース |
|2023.11| [StableLM(StabilityAI)](https://ja.stability.ai/blog/japanese-stable-lm-beta) | [HF](https://huggingface.co/stabilityai/japanese-stablelm-base-beta-70b) | 70B | | Llama2-70Bベース |
|2023.10| [LLM-jp](https://llm-jp.nii.ac.jp/blog/2024/02/09/v1.1-tuning.html) | [HF](https://huggingface.co/llm-jp) | 13B | DPO追加あり |
|| [name]() | [HF]() | | ||||

See more on [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm) and [日本語LLM評価](https://swallow-llm.github.io/evaluation/about.ja.html)


--- 

# Medical-Adaptation 


## Model
|When? | Name |  HF?  | Size | License | pretraining | finetuning/continual | test | misc.|
|---|---|---|---|---|---|---|---|---|
|2024.7| [Llama3-Preferred-MedSwallow]() | [HF](https://huggingface.co/pfnet/Llama3-Preferred-MedSwallow-70B) | 70B | Llama3 | Llama3 |   |  | japanese |
|2024.7| [Med42-v2]() | [HF](https://huggingface.co/m42-health/Llama3-Med42-70B) | 8,70B | Llama3 | llama3 |  ~1B tokens, including medical flashcards, exam questions, and open-domain dialogues. |  | |
|2024.7| [JMedLLM-v1]() | [HF](https://huggingface.co/stardust-coder/jmedllm-7b-v1) | 7B | qwen | Qwen2 |   |  | japanese |
|2024.6| [MedSwallow]() | [HF](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 70B | cc-by-nc-sa | Swallow |   |  | japanese |
|2024.5| [MMed-LLama3-8B(上海交通大学)](https://github.com/MAGIC-AI4Med/MMedLM) | [HF](https://huggingface.co/Henrychur/MMed-Llama-3-8B) | 8B | cc-by-sa | Llama3 |   |  |  |
|2024.5| [medX(JiviAI)]() | [HF](https://huggingface.co/jiviai/medX_v1) | 8B | Apache-2.0 | Llama3 |  100,000+ data, [ORPO](https://huggingface.co/blog/mlabonne/orpo-llama-3) |  |  |
|2024.4| [UltraMedical(TsinghuaC3I)](https://arxiv.org/html/2406.03949v1) | [HF](https://huggingface.co/TsinghuaC3I) | 8B | - | Llama3 |  | | |
|2024.4| [Meditron(EPFL)](https://www.meditron.io) | - | 8B | - | Llama3 |  | MedQA, MedMCQA, PubmedQA | SOTA |
|2024.4| [OpenBioLLM-70B]() | [HF](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B) | ? | - |  | |  | SOTA |
|2024.4| [Med-Gemini(Google)](https://arxiv.org/pdf/2404.18416) | closed | ? | - | Gemini | | |multimodal|
|2024.4| [Hippocrates](https://cyberiada.github.io/Hippocrates/) | [HF]() | 7B | |  | | |  | |
|2024.3| [AdaptLLM(Microsoft Research)](https://github.com/microsoft/LMOps/tree/main/adaptllm) | [HF](https://huggingface.co/AdaptLLM/medicine-LLM-13B) | 7B, 13B | | reading comprehensive corpora | | |  | ICLR2024 |
|2024.3| [Apollo](https://github.com/FreedomIntelligence/Apollo) | [HF](https://huggingface.co/FreedomIntelligence/Apollo-7B) | ~7B | | | | |  | multilingual |
|2024.2| [BiMediX](https://arxiv.org/pdf/2402.13253) | [HF](https://huggingface.co/BiMediX) | non-commercial | 8x7B | mixtral8x7B | | | MoE |
|2024.2| [Health-LLM(Rutgersなど)](https://arxiv.org/pdf/2402.00746.pdf) | | | | | | | RAG |
|2024.2| [BioMistral](https://arxiv.org/pdf/2402.10373.pdf) | [HF](https://huggingface.co/BioMistral) | 7B | - |  |  |  | | 
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

See also 
- [Awesome-Healthcare-Foundation-Models](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)
- [MedLLMsPracticalGuide](https://github.com/AI-in-Health/MedLLMsPracticalGuide).
- [医療分野に特化したLLM紹介](https://speakerdeck.com/stardust11)


## Evaluation
- [医療ドメイン特化LLMの性能はどうやって評価する？](https://zenn.dev/hellorusk/articles/04a29974138c7b)
- [MIRAGE Leaderboard](https://teddy-xionggz.github.io/MIRAGE/)
- [Japanese Medical Language Model Evaluation Harness](https://github.com/stardust-coder/japanese-lm-med-harness)
- [Open Medical LLM leaderboard](https://huggingface.co/blog/leaderboard-medicalllm), [leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
- [MMedBench](https://github.com/MAGIC-AI4Med/MMedLM)
- [MedEval](https://arxiv.org/pdf/2310.14088)

## Dataset

For Japanese medical dataset, see [JMedData4LLM](https://github.com/stardust-coder/jmed-data-for-llm).

Only Text
- [MedQA (USMLE)](https://github.com/jind11/MedQA) 
- [MedMCQA](https://arxiv.org/abs/2203.14371)
- [PubMedQA](https://arxiv.org/abs/1909.06146)
- [PubHealth](https://github.com/neemakot/Health-Fact-Checking)
- [MMLU](https://github.com/hendrycks/test) : includes medicine and other related fields(clinical topics covering clinical knowledge,
college biology, college medicine, medical genetics, professional medicine and anatomy)
- [JMMLU](https://github.com/nlp-waseda/JMMLU) : Japanese-translated version of MMLU
- [IgakuQA（Japanese National Medical License Exam）](https://github.com/jungokasai/IgakuQA)
- [J-ResearchCorpus](https://huggingface.co/datasets/kunishou/J-ResearchCorpus)
- [Apollo Corpus JP](https://huggingface.co/datasets/kunishou/ApolloCorpus-ja)
- HeadQA : Spanish healthcare system
- [K-Q&A](https://github.com/Itaymanes/K-QA)

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
- [MIMIC-ECG-IV](https://physionet.org/content/mimic-iv-ecg/) : ECG-caption dataset
- [ECG-QA](https://github.com/Jwoo5/ecg-qa)

Curations
- [Clinical NLP 2023](https://clinical-nlp.github.io/2023/resources.html)

See more on [He et al.(2023)](https://arxiv.org/pdf/2310.05694.pdf).