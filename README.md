# Keeping up with the latest LLMs !

* 2023.12 MistralAIから[Mixtral-8x7B](https://github.com/open-compass/MixtralKit)がリリースされました.
* 2023.12 [日本語LLMの学習データを問題視する記事](https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis)が公開されました.


## English-centric

|When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2023.12| [Amber](https://www.llm360.ai) | [HF](https://huggingface.co/LLM360/Amber) | 7B | apache-2.0 | Llama| totally open|
|2023.12| [Mixtral-8x7B](https://github.com/open-compass/MixtralKit)| [HF](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 8x7B | apache-2.0 ||mixture of ecperts|
|2023.12| [Mamba](https://github.com/state-spaces/mamba)| [HF](https://huggingface.co/state-spaces/mamba-2.8b) | 2.8B | apache-2.0 | based on state space model| | 
|2023.11| [Orca2(Microsoft)]() | [HF](https://huggingface.co/microsoft/Orca-2-13b) | 13B | MSRA-license| based on Llama2|||
|2023.11| [QWen(Alibaba)](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen/Qwen-72B) | 72B | [license](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)| 3T tokens | |beats Llama2|
|2023.9| [XWin](https://github.com/Xwin-LM/Xwin-LM) | [HF](https://huggingface.co/Xwin-LM/Xwin-LM-70B-V0.1)  | 70B | Llama2 |based on Llama2| also codes and math|
|2023.7| [Llama2(Meta)](https://ai.meta.com/llama/) | [HF](https://huggingface.co/meta-llama) | 70B | Llama2 |based on Llama2| chat-hf seems the best|
|2023.9| [Phi-1.5(Microsoft)](https://arxiv.org/abs/2309.05463) | [HF](https://huggingface.co/microsoft/phi-1_5) | 1.3B| MSRA-license||textbooks|
|| [name]() | [HF]() | | |||


PaLM(540B), GPT-4 are not open.


## Japanese-centric

|When? | Name |  HF?  | Size | License | pretraining | finetuning | misc.|
|---|---|---|---|---|---|---|---|
|2023.11| [StableLM(StabilityAI)](https://ja.stability.ai/blog/japanese-stable-lm-beta) | [HF](https://huggingface.co/stabilityai/japanese-stablelm-base-beta-70b) | 70B | |Llama2-70Bベース |
|2023.10| [LLM-jp]() | [HF](https://huggingface.co/llm-jp) | 13B | |
|| [name]() | [HF]() | | |

See more on [awesome-japanese-llm](https://github.com/llm-jp/awesome-japanese-llm)

## Medical

|When? | Name |  HF?  | Size | License | pretraining | finetuning | test | misc.|
|---|---|---|---|---|---|---|---|---|
|2023.12| [Medprompt(Microsoft)]() | not open | - | - | GPT-4 | ||multi-modal| 
|2023.7| [Med-Flamingo](https://github.com/snap-stanford/med-flamingo) | [HF]() | 8.3B| ? | OpenFlamingo | MTB | Visual USMLE|based on Flamingo |
|2023.7| [LLaVa-Med(Microsoft)](https://github.com/microsoft/LLaVA-Med) | [HF](https://huggingface.co/microsoft/llava-med-7b-delta) | 13B | - | LLaVa| medical dataset |VAQ-RAD, SLAKE, PathVQA|multi-modal| 
|2023| [Med-PaLM M(Google)](https://arxiv.org/abs/2307.14334) | not open | | - | PaLM2 | | |multi-modal| 
|2023.5| [Med-PaLM2(Google)](https://arxiv.org/abs/2305.09617) | not open |340B | - | PaLM2 | | |
|2022.12| [Med-PaLM(Google)](https://arxiv.org/abs/2212.13138) | not open | 540B| - | PaLM | | | |
|| [name]() | [HF]() | | |


### Lists of train dataset
- MTB: chopped cleaned text and images collected from 4721 textbooks.
- PMC-15M : the largest biomedical image-text dataset
- PMC-OA : 1.6M image-caption pairs

### Lists of test dataset
- VQA-RAD : 
- SLAKE : 
- PathVQA : 
- Visual USMLE : 618 USMLE-style QA
- [MedVTE](https://github.com/ynklab/MedVTE):numeric understanding
- [MedICCaT](https://github.com/allenai/medicat):image, caption, textual reference

See more on [He et al.(2023)](https://arxiv.org/pdf/2310.05694.pdf).