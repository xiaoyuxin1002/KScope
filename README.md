# KScope: A Framework for Characterizing the Knowledge Status of Language Models

In this paper, we propose a taxonomy of five knowledge statuses and introduce KScope, a hierarchical testing framework to characterize LLM knowledge status.  
We apply KScope to nine LLMs across four datasets and investigate three questions: 
1. How does context update LLMs' knowledge status?
2. What context features drive the desired knowledge update?
3. What context augmentations work best across knowledge statuses?

**Please note that we are unable to upload the datasets and model outputs due to size constraints.**  
**However, all results can be easily reproduced using the provided code.**

<hr style="border: none; height: 1px; solid #eaecef;" />

We recommend the following workflow:
1. Dataset Preparation: Run ```1-Preparation.ipynb``` to prepare the four evaluation datasets.
2. Hyperparameter Search: Run ```2-Hyperparameter.ipynb``` to determine the number of question paraphrases and sample responses needed for consistent characterization of LLM knowledge status.
3. Model Response: Run ```response.sh``` to collect the responses of the nine LLMs across the four datasets.
4. Knowledge Status Characterization (Q1): Run ```status.sh``` to apply KScope to the collected model responses, and ```3-Status.ipynb``` to investigate context-induced shifts in LLM knowledge status.
5. Feature Importance Analysis (Q2): Run ```uncertainty.sh``` to compute models' perplexity and entropy, ```4-Feature.ipynb``` to compute the other features, and ```5-Analysis.ipynb``` to analyze the results.
6. Context Augmentation Strategies (Q3): Run ```6-GPT.ipynb``` to collect GPT-4o's responses, and ```7-Augmentation.ipynb``` to evaluate the effectiveness of different augmentation strategies.