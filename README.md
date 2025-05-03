# 最新LLM & 医療LLM 一覧

LLMの最新情報のまとめ.

Awesome latest LLMs. Let's keep up with the latest LLMs and medical LLMs!

**NEWS**
- 🔥2025.5 Ai2からOLMo2-1Bの最新版がリリースされました！
- 🔥2025.4 Qwen3がリリースされました！
- 🔥2025.4 Llama4がリリースされました！

<details>

<summary>History</summary>
- 2025.3 🔥Gemma3がリリースされました！
- 2025.3 🔥Llama-Swallowの最新版がリリースされました！
- 2025.3 🔥Sarashina2.2がリリースされました.
- 2025.2 小規模言語モデル（SLM）の特集を始めました
- 2025.2 🔥 Grok3がxAIから発表されました！
- 2025.2 モデルの絞り込みを行いました.
- 2025.1 🔥 [Qwen2.5-Max](https://qwenlm.github.io/blog/qwen2.5-max/)がリリースされました. モデル非公開.
- 2025.1 🔥Minimax-01がリリースされました！
- 2024.11 🔥Qwenチームからreasoningに優れたとされる実験的モデルQWQがリリースされました！
- 2024.7 🔥東工大からLlama3の日本語継続学習モデルが発表！
- 2024.6 🔥ELYZAからLlama3の日本語継続学習モデルが発表！
- 2024.6 🔥Googleから27BのGemma2が公開！何が強みか教えて！
- 2024.6 🔥NVIDIAが340Bの巨大モデルを公開！publicにしては最大級
- 2024.6 🔥Qwen2シリーズが登場！日本語も優秀！
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

## Table of Contents
[English-centric](#english-centric)  
[Japanese-centric](#japanese-centric)  
[Medical Adaptation](#medical-adaptation)  

## English-centric

| When? | Name |  HF?  | Size(max) | License | pretraining/base | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2025.4| [Qwen3 (Alibaba)]()|[HF](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)| 0.6~235B |apache-2.0|| |
|2025.4| [Llama4 (Meta)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)|[HF](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164)|17B|llama4|30T token||10M token|
|2025.1| [DeepSeek-V3-0324]() | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) | 671B | [link](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL) | 14.8T |  | MoE(37B) |
|2025.3| [Gemma3 (Google)]() | [HF](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)| 27B | [gemma](https://ai.google.dev/gemma/terms) | | |
|2025.1| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1)| 671B | MIT | | |
|2024.12| [Llama3.3 (Meta)]() | [HF](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000) | 70B | [llama3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE) | |  |  |
|2024.12| [Phi-4 (Microsoft)](https://www.microsoft.com/en-us/research/uploads/prod/2024/12/P4TechReport.pdf) |[HF](https://huggingface.co/NyxKrage/Microsoft_Phi-4) | 14B | msrla |  |  | small, sft, dpo |
|2024.11| [QWQ (Alibaba)]() |[HF](https://huggingface.co/Qwen/QwQ-32B-Preview) | 32B | apache-2.0 | Qwen2.5 |  |  reasoning |
|2024.9| [Qwen 2.5 (Alibaba)]() |[HF](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) | 0.5~72B | apache2.0 |  |  | [long context available](https://huggingface.co/collections/Qwen/qwen25-1m-679325716327ec07860530ba) |
<!-- 
|2024.12| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) | 671B | [link](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL) | 14.8T | sft, RL | MoE |
|2024.7| [Reflection]() |[HF](https://huggingface.co/mattshumer/ref_70_e3) | 70B | llama3.1 | Llama 3.1 | synthetic data (Glaive) |  |
|2025.1| [InternLM v3]() | [HF](https://huggingface.co/internlm/internlm3-8b-instruct) | 8B | apache-2.0 | 4T token |  | deep thinking | 
|2024.10| [Llama3.2(Meta)]() |[HF](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | 1B,3B | llama3.2 |  llama3.2 |  |  |
|2024.7| [Llama3.1(Meta)]() |[HF]() | 70B, 405B | Llama3.1 |  |  |  |
|2024.6| [Gemma2(Google)]() |[HF](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | 2B, 9B, 27B | gemma |  |  |  |
|2024.6| [Nemotron(NVIDIA)]() |[HF](https://huggingface.co/nvidia/Nemotron-4-340B-Instruct) | 340B |  | - | - |  |
|2024.6| [Qwen2(Alibaba)]() |[HF](https://huggingface.co/Qwen/Qwen2-72B) | 7~72B | tongyi-qianwen | - | - |  |
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
|2023.7| [Llama2(Meta)](https://ai.meta.com/llama/) | [HF](https://huggingface.co/meta-llama) | 70B | Llama2 | 2T tokens| chat-hf seems the best| -->

<!-- |2024.1| [LLaMa-Pro-8B(Tencent)]() | [HF](https://huggingface.co/TencentARC/LLaMA-Pro-8B) | 8B | Llama2 ||| -->
<!-- |2023.12| [Amber](https://www.llm360.ai) | [HF](https://huggingface.co/LLM360/Amber) | 7B | apache-2.0 | Llama|| totally open| -->
<!-- |2023.11| [Orca2(Microsoft)]() | [HF](https://huggingface.co/microsoft/Orca-2-13b) | 13B | MSRA-license| based on Llama2||| -->
<!-- |2023.9| [Phi-1.5(Microsoft)](https://arxiv.org/abs/2309.05463) | [HF](https://huggingface.co/microsoft/phi-1_5) | 1.3B| MSRA-license||textbooks| -->
- PaLM(540B), PaLM2(340B) and GPT-4 are not open.
- See also [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

## Japanese-centric

| When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2024.3| [Stockmark 2]() | [HF](https://huggingface.co/stockmark/Stockmark-2-100B-Instruct-beta) | 100B |  |  | |  |
|2024.3| [Llama-3.3-Swallow-70B-Instruct-v0.4]() | [HF](https://huggingface.co/tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4) | 70B | [llama3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/LICENSE) | Llama3.3 | | JMT-Bench 0.772 |
|2025.2| [PlaMo 2 (PFN)](https://tech.preferred.jp/ja/blog/plamo-2-8b/) | [HF](https://huggingface.co/pfnet/plamo-2-8b) | 8B | [plamo](https://tech.preferred.jp/ja/blog/plamo-community-license/) ||| Samba |
|2025.3| [Bakeneko (rinna)]()| [HF](https://huggingface.co/collections/rinna/qwen25-bakeneko-67aa2ef444910bbc55a21222) [HF](https://huggingface.co/rinna/qwq-bakeneko-32b) | 32B | apache-2.0 |  |  | |
|2025.1| [DeepSeek-R1-Distil-Qwen-Japanese(CyberAgent)]()| [HF](https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese) | 32B | MIT | [distil](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | Japanese dataset | |
|2024.12| [llm-jp-3-172b-instruct3]() | [HF](https://huggingface.co/llm-jp/llm-jp-3-172b-instruct3) | 172B | [利用規約](https://huggingface.co/llm-jp/llm-jp-3-172b-instruct3/raw/main/LICENSE_ja)  |  | |  |
<!--|2024.12| [Llama-3.1-Swallow-70B-Instruct-v0.3]() | [HF](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3) | 70B | [llama3.1](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/LICENSE) | Llama3.1 | |  |
|2024.7| [Llama-3.1-70B-Japanese-Instruct-2407]() | [HF](https://huggingface.co/cyberagent/Llama-3.1-70B-Japanese-Instruct-2407) | 70B | Llama3.1 | Llama3.1  | |  | -->
<!-- |2024.7| [LLama3-Swallow]() | [HF](https://huggingface.co/tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1) | 70B | Llama3 | Llama3  | |  |
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
|2023.10| [LLM-jp](https://llm-jp.nii.ac.jp/blog/2024/02/09/v1.1-tuning.html) | [HF](https://huggingface.co/llm-jp) | 13B | DPO追加あり | -->

- See more on [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm) and [日本語LLM評価](https://swallow-llm.github.io/evaluation/about.ja.html)
- Let's go soverign AI!

## Small language models (SLM)
| When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2025.5| [OLMo-2](https://huggingface.co/allenai) | [HF](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) | 1B | | | |
|2025.3| [Gemma3]() | [HF](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)| 1B,4B | [gemma](https://ai.google.dev/gemma/terms) | | |
|2025.3| [Sarashina2.2](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1) |  | 0.5B,1B,3B | mit | | | ELYZA-tasks=3.75 |
|2025.2 | [SmolLM(huggingface)](https://github.com/huggingface/smollm) | [HF](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)| 135M~1.7B| apache-2.0 | | 
|2025.2| [Phi-4 mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) |  | 3.8B | mit | | |  |
|2025.2| [PLaMo 2 2B](https://tech.preferred.jp/ja/blog/plamo-2-2b/) | None | 2B | apache-2.0| |  | pruning, tested on HumanEval+ |
|2025.2| [PLaMo 2 1B]() | [HF](https://huggingface.co/pfnet/plamo-2-1b) | 1B | apache-2.0| 4T (1.25T tokens Japanese)| | base model only |
|2025.1| [TinySwallow-1.5B-Instruct]() | [HF](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct) | 1.5B | apache-2.0 | qwen | Japanese | TAID from Qwen2.5-32B|
|2024.9| [Qwen2.5]() |[HF](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) | 0.5,1.5,3B | apache2.0 |  |  |  |
|2024.8| [Phi-3.5-mini-instruct]() |[HF](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | apache2.0 | MIT |  |  |


--- 

# Medical-Adaptation 


## Model
|When? | Name |  HF?  | Size | License | pretraining | finetuning/continual | test | misc.|
|---|---|---|---|---|---|---|---|---|
|2025.4| [OmniV-Med(Alibaba)]() | | 1.5,7B |  | 252K instruction data|    | 11 benchmarks (2D/3D image and video) |  |
|2025.4| [JPharmatron(EQUES)]() | [HF]() | 7B |  | Qwen2.5 | pharma corpus   | None | japanese |
|2025.2| [Preferred-MedLLM-Qwen-72B]() | [HF](https://huggingface.co/pfnet/Preferred-MedLLM-Qwen-72B) | 72B | Qwen | Qwen2.5 | original corpus   | IgakuQA | japanese |
|2025.2| [OpenMeditron](https://huggingface.co/OpenMeditron)|[HF](https://huggingface.co/OpenMeditron/Meditron3-70B) | 7~70B | |||MedQA etc. | 
|2025.1| [Huatuo-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)|[HF](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-72B) | 72B | apache-2.0 | 
|2024.8| [LLaVA-Med++](https://github.com/UCSC-VLAA/MedTrinity-25M) | [HF](https://huggingface.co/MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct-FT-S2) | 8B | ? | MedTrinity-25M | VQA-RAD etc. |   |  | |
|2024.7| [MedLlama3-JP (EQUES)]() | [HF](https://huggingface.co/EQUES/MedLLama3-JP-v2) | 8B | Llama3 | Llama3 |   |  | japanese, merge model|
|2024.7| [Llama3-Preferred-MedSwallow]() | [HF](https://huggingface.co/pfnet/Llama3-Preferred-MedSwallow-70B) | 70B | Llama3 | Llama3 |   |  | japanese |
|2024.7| [Med42-v2]() | [HF](https://huggingface.co/m42-health/Llama3-Med42-70B) | 8,70B | Llama3 | llama3 |  ~1B tokens, including medical flashcards, exam questions, and open-domain dialogues. |  | |
|2024.7| [JMedLLM-v1]() | [HF](https://huggingface.co/stardust-coder/jmedllm-7b-v1) | 7B | qwen | Qwen2 |   |  | japanese |
|2024.6| [MedSwallow]() | [HF](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 70B | cc-by-nc-sa | Swallow |   |  | japanese |
|2024.5| [MMed-LLama3-8B(上海交通大学)](https://github.com/MAGIC-AI4Med/MMedLM) | [HF](https://huggingface.co/Henrychur/MMed-Llama-3-8B) | 8B | cc-by-sa | Llama3 |   |  |  |
|2024.5| [medX(JiviAI)]() | [HF](https://huggingface.co/jiviai/medX_v1) | 8B | Apache-2.0 | Llama3 |  100,000+ data, [ORPO](https://huggingface.co/blog/mlabonne/orpo-llama-3) |  |  |
|2024.4| [UltraMedical(TsinghuaC3I)](https://arxiv.org/html/2406.03949v1) | [HF](https://huggingface.co/TsinghuaC3I) | 8B | - | Llama3 |  | | |
|2024.4| [Meditron(EPFL)](https://www.meditron.io) | - | 8B | - | Llama3 |  | MedQA, MedMCQA, PubmedQA | SOTA |
|2024.4| [OpenBioLLM]() | [HF](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B) | 8, 70B |  | Llama3 | |  | SOTA |
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
|2023.5| [Almanac(Stanford)](https://arxiv.org/pdf/2303.01229.pdf)| ? | ? | text-davinci-003 |  | | RAG |
|2023.5| [Med-PaLM2(Google)](https://arxiv.org/abs/2305.09617) | not open | 340B | - | PaLM2 | | |
|2022.12| [Med-PaLM(Google)](https://arxiv.org/abs/2212.13138) | not open | 540B| - | PaLM | | | |
|| [name]() | [HF]() | | |

See also 
- [Awesome-Healthcare-Foundation-Models](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)
- [Awesome-Medical-Large-Language-Models](https://github.com/burglarhobbit/Awesome-Medical-Large-Language-Models)
- [Awesome-Medical-LLM](https://github.com/BARUDA-AI/Awesome-Medical-LLM)
- [MedLLMsPracticalGuide](https://github.com/AI-in-Health/MedLLMsPracticalGuide).
- [医療分野に特化したLLM紹介](https://speakerdeck.com/stardust11)


## Evaluation benchmarks

- [MIRAGE Leaderboard](https://teddy-xionggz.github.io/MIRAGE/)
- [Open Medical LLM leaderboard](https://huggingface.co/blog/leaderboard-medicalllm), [leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
- [MMedBench](https://github.com/MAGIC-AI4Med/MMedLM)
- [MedEval](https://arxiv.org/pdf/2310.14088)
- [MEDIC](https://arxiv.org/pdf/2409.07314)

## Dataset

For Japanese medical dataset, see [JMedData4LLM](https://github.com/stardust-coder/jmed-data-for-llm).

### Only Text

**Representative medical benchmarks**
- [MedQA (USMLE)](https://github.com/jind11/MedQA) 
- [MedMCQA](https://arxiv.org/abs/2203.14371)
- [PubMedQA](https://arxiv.org/abs/1909.06146)
- [MMLU]() : subset related to medicine is often used.

Dataset from [**FreedomIntelligence**](https://github.com/FreedomIntelligence)
- [Medical O1 Reasoning](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [Medical O1 Verifiable Problem](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem)
- [Disease Database](https://huggingface.co/datasets/FreedomIntelligence/Disease_Database)
- etc.

Dataset from [**OnDeviceMedNotes**](https://huggingface.co/OnDeviceMedNotes)
- [synthetic-medical-conversations-deepseek-v3](https://huggingface.co/datasets/OnDeviceMedNotes/synthetic-medical-conversations-deepseek-v3)

**Others**
- [HealthsearchQA]() : 3173 samples, used in MedPaLM paper
- [LiveQA]() : 634+10, used in MedPaLM paper
- [PubHealth](https://github.com/neemakot/Health-Fact-Checking)
- [MMLU](https://github.com/hendrycks/test) : includes medicine and other related fields(clinical topics covering clinical knowledge,
college biology, college medicine, medical genetics, professional medicine and anatomy)
- [HeadQA](https://huggingface.co/datasets/dvilares/head_qa) : Spanish healthcare system
- [K-Q&A](https://github.com/Itaymanes/K-QA)
- Clincal Case Challenges : NEHM dataset and JAMA dataset
- MeDiSumQA : discharge summaries from the MIMIC-IV
- MeDiSumCode : ICD-10 codes
- MedNLI : MIMIC-III dataset, logical relationship between a premise and a hypothesis
- MeQSum : summarizing health queries
- [LongHealth](https://github.com/kbressem/LongHealth) : 20 patient records, answer questions about them from a long document.
- [Medical Eval Sphere](https://github.com/lavita-ai/medical-eval-sphere) : Long form medical questions
- [MedCalcBench](https://github.com/ncbi-nlp/MedCalc-Bench):  [HF](https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.0)
- [PMC Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients/tree/main)
- [MedQA-Calc](https://huggingface.co/datasets/Nicholas-Wan/MedQA-Calc)
- [MedS-Bench](https://huggingface.co/datasets/Henrychur/MedS-Bench)
- [MedQuAD](https://github.com/abachaa/MedQuAD)
- [TJH Dataset](https://github.com/HAIRLAB/Pre_Surv_COVID_19)
- [MIMIC-IV]() : Sourced from the EHRs of the Beth Israel Deaconess Medical Center.
- [ClinicBench]() : 17 comprehensive benchmarks. 
- [EquityMedQA](https://huggingface.co/datasets/katielink/EquityMedQA) : Open-ended Q&A for equity and bias mitigation.
- [MedDistractQA](https://huggingface.co/datasets/KrithikV/MedDistractQA)


### Image + Text
- MTB: chopped cleaned text and images collected from 4721 textbooks.
- PMC-15M : the largest biomedical image-text dataset
- PMC-OA : 1.6M image-caption pairs
- [MedICaT](https://github.com/allenai/medicat): image, caption, textual reference
- [VQA-RAD](https://osf.io/89kps/) : 3515 question–answer pairs on 315 radiology images.
- SLAKE : bilingual dataset (English&Chinese) consisting of 642 images and 14,028 question-answer pairs
- PathVQA : pathology image + caption
- Visual USMLE : 618 USMLE-style QA
- [MedVTE](https://github.com/ynklab/MedVTE): numeric understanding
- [MedAlign(Stanford)](https://github.com/som-shahlab/medalign)
- [MIMIC-ECG-IV](https://physionet.org/content/mimic-iv-ecg/) : ECG-caption dataset
- [ECG-QA](https://github.com/Jwoo5/ecg-qa)
- [MedEval](https://github.com/ZexueHe/MedEval)
- [MedTrinity](https://github.com/UCSC-VLAA/MedTrinity-25M)
- [Clinical NLP 2023](https://clinical-nlp.github.io/2023/resources.html)
- [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA)

See more on 
- [He et al.(2023)](https://arxiv.org/pdf/2310.05694.pdf)
- [MedLLMsPracticalGuide](https://github.com/AI-in-Health/MedLLMsPracticalGuide?tab=readme-ov-file#-practical-guide-for-medical-data)
- [Medical datasets for LLMs (collection)](https://huggingface.co/collections/mfmezger/medical-datasets-for-llms-66bbb4e371cc405bc4d6b28a)