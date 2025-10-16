# KScope: A Framework for Characterizing the Knowledge Status of Language Models

In this paper, we propose a taxonomy of five knowledge statuses and introduce KScope, a hierarchical testing framework to characterize LLM knowledge status.  
We apply KScope to nine LLMs across four datasets and investigate three questions: 
1. How does context update LLMs' knowledge status?
2. What context features drive the desired knowledge update?
3. What context augmentations work best across knowledge statuses?

**Please note that we are unable to upload the datasets and model outputs due to size constraints.**  
**However, all results can be easily reproduced using the provided code.**

<hr style="border: none; height: 1px; solid #eaecef;" />

### Recommended Workflow

1. Dataset Preparation:
   - Run ```1-Preparation.ipynb``` to prepare the four evaluation datasets.
2. Hyperparameter Search:
   - Run ```2-Hyperparameter.ipynb``` to determine the number of question paraphrases and sample responses required for consistent characterization of LLM knowledge status.
3. Model Response Collection:
   - Run ```response.sh``` to collect responses from the nine LLMs on the four datasets in the multi-choice setting with gold context.
   - To test with noisy context, replace "evidence" with "retrieved" in ```response.py```.
   - To switch to the open-ended setting, use the provided commented instruction prompts in ```response.py```.
4. Knowledge Status Characterization (Q1):
   - Apply KScope to the collected responses to characterize LLM knowledge status:
   - Run ```status.sh``` for the multi-choice setting (with either gold or noisy context).
   - Run ```status_open.sh``` for the open-ended setting (with gold context).
   - Use ```3-Status.ipynb``` and ```4-Settings.ipynb``` to further explore and visualize context-induced shifts in knowledge status.
5. Feature Importance Analysis (Q2):
   - Run ```uncertainty.sh``` to compute perplexity and entropy.
   - Run ```5-Feature.ipynb``` to compute additional features.
   - Run ```6-Analysis.ipynb``` to analyze feature importance.
6. Context Augmentation Strategies (Q3):
   - Run ```7-GPT.ipynb``` to collect GPT-4o responses.
   - Run ```8-Augmentation.ipynb``` to evaluate the effectiveness of different augmentation strategies.