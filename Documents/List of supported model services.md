# 模型支持列表 | List of supported model services

|Architecture|Models|Example HF Models|LoRA|PP|
|---|---|---|---|---|
|AquilaForCausalLM|Aquila, Aquila2|BAAI/Aquila - 7B, BAAI/AquilaChat - 7B, etc.|√︎|√︎
|ArcticForCausalLM|Arctic|Snowflake/snowflake - arctic - base, Snowflake/snowflake - arctic - instruct, etc.||√︎|
|BaiChuanForCausalLM|Baichuan2, Baichuan|baichuan - inc/Baichuan2 - 13B - Chat, baichuan - inc/Baichuan - 7B, etc.|√︎|√︎|
BloomForCausalLM|BLOOM, BLOOMZ, BLOOMChat|bigscience/bloom, bigscience/bloomz, etc.||√︎|
BartForConditionalGeneration|BART|facebook/bart - base, facebook/bart - large - cnn, etc.|||
|ChatGLMModel|ChatGLM|THUDM/chatglm2 - 6b, THUDM/chatglm3 - 6b, etc.|√︎|√︎|
|CohereForCausalLM, Cohere2ForCausalLM|Command - R|CohereForAI/c4ai - command - r - v01, CohereForAI/c4ai - command - r7b - 12 - 2024, etc.|√︎|√︎|
|DbrxForCausalLM|DBRX|databricks/dbrx - base, databricks/dbrx - instruct, etc.||√︎|
|DeciLMForCausalLM|DeciLM|Deci/DeciLM - 7B, Deci/DeciLM - 7B - instruct, etc.||√︎|
|DeepseekForCausalLM|DeepSeek|deepseek - ai/deepseek - llm - 67b - base, deepseek - ai/deepseek - llm - 7b - chat etc.||√︎|
|DeepseekV2ForCausalLM|DeepSeek - V2|deepseek - ai/DeepSeek - V2, deepseek - ai/DeepSeek - V2 - Chat etc.||√︎|
|Deepseek3ForCausalLM|DeepSeek - V3|deepseek - ai/DeepSeek - V3 - Base, deepseek - ai/DeepSeek - V3 etc.||√︎|
|ExaoneForCausalLM|EXAONE - 3|LGAI - EXAONE/EXAONE - 3.0 - 7.8B - Instruct, etc.|√︎|√︎|
|FalconForCausalLM|Falcon|tiiuae/falcon - 7b, tiiuae/falcon - 40b, tiiuae/falcon - rw - 7b, etc.||√︎|
|FalconMambaForCausalLM|FalconMamba|tiiuae/falcon - mamba - 7b, tiiuae/falcon - mamba - 7b - instruct, etc.|√︎|√︎|
|GemmaForCausalLM|Gemma|google/gemma - 2b, google/gemma - 7b, etc.|√︎|√︎|
|Gemma2ForCausalLM|Gemma2|google/gemma - 2 - 9b, google/gemma - 2 - 27b, etc.|√︎|√︎|
|GlmForCausalLM|GLM - 4|THUDM/glm - 4 - 9b - chat - hf, etc.|√︎|√︎|
|GPT2LMHeadModel|gpt2|gpt2, gpt2 - xl, etc.||√︎|
|GPTBigCodeForCausalLM|StarCoder, SantaCoder, WizardCoder|bigcode/starcoder, bigcode/gpt_bigcode - santacoder, WizardLM/WizardCoder - 15B - V1.0, etc.|√︎|√︎|
|GPTJForCausalLM|GPT - J|EleutherAI/gpt - j - 6b, nomic - ai/gpt4all - j, etc.||√︎|
|GPTNeoXForCausalLM|GPT - NeoX, Pythia, OpenAssistant, Dolly V2, StableLM|EleutherAI/gpt - neox - 20b, EleutherAI/pythia - 12b, OpenAssistant/oasst - sft - 4 - pythia - 12b - epoch - 3.5, databricks/dolly - v2 - 12b, stabilityai/stablelm - tuned - alpha - 7b, etc.||√︎|
|GraniteForCausalLM|Granite 3.0, Granite 3.1, PowerLM|ibm - granite/granite - 3.0 - 2b - base, ibm - granite/granite - 3.1 - 8b - instruct, ibm/PowerLM - 3b, etc.|√︎|√︎|
|GraniteMoeForCausalLM|Granite 3.0 MoE, PowerMoE|ibm - granite/granite - 3.0 - 1b - a400m - base, ibm - granite/granite - 3.0 - 3b - a800m - instruct, ibm/PowerMoE - 3b, etc.|√︎|√︎|
|GritLM|GritLM|parasail - ai/GritLM - 7B - vllm.|√︎|√︎|
|InternLMForCausalLM|InternLM|Internlm/internlm - 7b, internlm/internlm - chat - 7b, etc.|√︎|√︎|
|InternLM2ForCausalLM|InternLM2|internlm/internlm2 - 7b, internlm/internlm2 - chat - 7b, etc.|√︎|√︎|
|InternLM3ForCausalLM|InternLM3|internlm/internlm3 - 8b - instruct, etc.|√︎|√︎|
|JAISLMHeadModel|Jais|inceptionai/jais - 13b, inceptionai/jais - 13b - chat, inceptionai/jais - 30b - v3, inceptionai/jais - 30b - chat - v3, etc.||√︎|
|JambaForCausalLM|Jamba|ai21labs/AI21 - Jamba - 1.5 - Large, ai21labs/AI21 - Jamba - 1.5 - Mini, ai21labs/Jamba - v0.1, etc.|√︎|√︎|
|LlamaForCausalLM|Llama 3.1, Llama 3,Llama 2, LLaMA, Yi|meta - llama/Meta - Llama - 3.1 - 405B - Instruct, meta - llama/Meta - Llama - 3.1 - 70B, meta - llama/Meta - Llama - 3 - 70B - Instruct, meta - llama/Llama - 2 - 70b - hf, 01 - ai/Yi - 34B, etc.|√︎|√︎|
|MambaForCausalLM|Mamba|state - spaces/mamba - 130m - hf, state - spaces/mamba - 790m - hf, state - spaces/mamba - 2.8b - hf, etc.||√︎|
|MiniCPMForCausalLM|MiniCPM|openbmb/MiniCPM - 2B - sft - bf16, openbmb/MiniCPM - 2B - dpo - bf16, openbmb/MiniCPM - S - 1B - sft, etc.|√︎|√︎|
|MiniCPM3ForCausalLM|MiniCPM3|openbmb/MiniCPM3 - 4B, etc.|√︎|√︎|
|MistralForCausalLM|Mistral, Mistral - Instruct|mistralai/Mistral - 7B - v0.1, mistralai/Mistral - 7B - Instruct - v0.1, etc.|√︎|√︎|
|MixtralForCausalLM|Mixtral - 8x7B, Mixtral - 8x7B - Instruct|mistralai/Mixtral - 8x7B - v0.1, mistralai/Mixtral - 8x7B - Instruct - v0.1, mistral - community/Mixtral - 8x22B - v0.1, etc.|√︎|√︎|
|MPTForCausalLM|MPT, MPT - Instruct, MPT - Chat,MPT - StoryWriter|mosaicml/mpt - 7b, mosaicml/mpt - 7b - storywriter, mosaicml/mpt - 30b, etc.||√︎|
|NemotronForCausalLM|Nemotron - 3, Nemotron - 4, Minitron|nvidia/Minitron - 8B - Base, mgoin/Nemotron - 4 - 340B - Base - hf - FP8, etc.|√︎|√︎|
|OLMoForCausalLM|OLMo|allenai/OLMo - 1B - hf, allenai/OLMo - 7B - hf, etc.||√︎|
|OLMo2ForCausalLM|OLMo2|allenai/OLMo2 - 7B - 1124, etc.||√︎|
|OLMoEForCausalLM|OLMoE|allenai/OLMoE - 1B - 7B - 0924, allenai/OLMoE - 1B - 7B - 0924 - Instruct, etc.|√︎|√︎|
|OPTForCausalLM|OPT, OPT - IML|facebook/opt - 66b, facebook/opt - iml - max - 30b, etc.||√︎|
|OrionForCausalLM|Orion|OrionStarAI/Orion - 14B - Base, Ori√StarAI/Orion - 14B - Chat, etc.||√︎|
|PhiForCausalLM|Phi|microsoft/phi - 1_5, microsoft/phi - 2, etc.|√︎|√︎|
|Phi3ForCausalLM|Phi - 4, Phi - 3|microsoft/Phi - 4, microsft/Phi - 3 - mini - 4k - instruct, microsoft/Phi - 3 - mini - 128k - instruct, microsoft/Phi - 3 - medium - 128k - instruct, etc.|√︎|√︎|
|Phi3SmallForCausalLM|Phi - 3 - Small|microsoft/Phi - 3 - small -8k - instruct, microsoft/Phi - 3 - small - 128k - instruct, etc.||√︎|
|PhiMoEForCausalLM|Phi - 3.5 - MoE|microsoft/Phi - 3.5 - MoE - instruct, etc.|√︎|√︎|
|PersimmonForCausalLM|Persimmon|adept/persimmon - 8b - base, adept/persimmon - 8b - chat, etc.||√︎|
|QWenLMHeadModel|Qwen|Qwen/Qwen - 7B, Qwen/Qwen - 7B - Chat, etc.|√︎|√︎|
|Qwen2ForCausalLM|QwQ, Qwen2|Qwen/QwQ - 32B - Preview, Qwen/Qwen2 - 7B - Instruct, Qwen/Qwen2 - 7B, etc.|√︎|√︎|
|Qwen2MoeForCausalLM|Qwen2MoE|Qwen/Qwen1.5 - MoE - A2.7B, Qwen/Qwen1.5 - MoE - A2.7B - Chat, etc.||√︎|
|StableLmForCausalLM|StableLM|stabilityai/stablelm - 3b - 4e1t, stabilityai/stablelm - base - alpha - 7b - v2, etc.||√︎|
|Starcoder2ForCausalLM|Starcoder2|bigcode/starcoder2 - 3b, bigcode/starcoder2 - 7b, bigcode/starcoder2 - 15b, etc.||√︎|
|SolarForCausalLM|Solar Pro|upstage/solar - pro - preview - instruct, etc.|√︎|√︎|
|TeleChat2ForCausalLM|TeleChat2|TeleAI/TeleChat2 - 3B, TeleAI/TeleChat2 - 7B, TeleAI/TeleChat2 - 35B, etc.|√︎|√︎|
|XverseForCausalLM|XVERSE|xverse/XVERSE - 7B - Chat, xverse/XVERSE - 13B - Chat,xverse/XVERSE - 65B - Chat, etc.|√︎|√︎|