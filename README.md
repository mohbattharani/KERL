# KERL


### Title: KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models


### Abstract:

Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis.


### Download Data: 

- Download [Recipe1M](http://im2recipe.csail.mit.edu), it may not be avaialble publically.
- FoodKG: Pleaes refer to [FoodKG](https://foodkg.github.io)


### Model Training 

- Provide parameters in `script/finetune.sh`
- RUN: 

      $bash script/finetune.sh 


### DEMO: 

This demo uses a subset of FoodKG for faster response to answer user questions.  

- Given base model path: `microsoft/Phi-3-mini-128k-instruct` or download it to local dir from: [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 

- Download adapters from [KERL-HF](https://huggingface.co/mohbattharani) and all adapters in the same dir such as `checkpoints` and within respective folders. 

      $python demo.py


change question as you wish. 



## Cite

```bibtex
@article{mohbat2024llavachef,
  title={KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models},
  author={Fnu Mohbat, Mohammed J. Zaki},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}

```
