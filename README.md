# News Summarization

This project presents a fine-tuned T5-Base model for automatic news summarization, utilizing the CNN/DailyMail dataset. The project showcases both the technical process and an interactive Gradio interface for end-user engagement, enabling effective exploration of text summarization using modern NLP techniques.

## Overview

### Objective

The primary objective of this project is to develop a summarization tool that can generate concise and accurate summaries for news articles using a fine-tuned T5-Base model. The tool demonstrates the potential of transformer-based models in text-to-text generation tasks and provides an interactive interface for users to experiment with the model.

### Key Features

1. Data Preprocessing: Handling large-scale datasets by shuffling and sampling, making them compatible with the T5-Base model requirements.
2. Model Fine-Tuning: Leveraging transfer learning to fine-tune a pre-trained T5-Base model on the CNN/DailyMail dataset.
3. Evaluation Metrics: Using ROUGE scores to assess summarization performance.
4. Visualization: Creating dynamic bar charts for input and summary word count comparisons, along with static performance metrics visualizations.
5. User Interaction: Providing an intuitive Gradio interface for inputting news articles and generating summaries.

## Model Card & Dataset Card

### Model
Name: T5-Base
Source: [google-t5/t5-base](https://huggingface.co/google-t5/t5-base)
Architecture: Encoder-Decoder Transformer
Usage: Text-to-text generation tasks, fine-tuned specifically for summarization tasks in this project.

### Dataset
Name: CNN/DailyMail
Source: [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
Purpose: News articles paired with human-written summaries, serving as a benchmark for text summarization tasks.
Processing:
Shuffled and sampled 10,000 rows for training and validation.
Processed into formats compatible with the T5-Base tokenizer and model.

## Critical Analysis

### Strengths

1. Efficiency: The T5-Base model demonstrates strong performance in generating concise and coherent summaries.
2. Scalability: Pre-trained transformer models like T5 allow fine-tuning with relatively small datasets to achieve domain-specific improvements.
3. Visualization: Performance metrics are clearly illustrated, making the model’s strengths and weaknesses accessible to non-technical audiences.

### Limitations

1. Generalization: While the model performs well on the CNN/DailyMail dataset, it may not generalize to all news domains without additional fine-tuning.
2. Length Constraints: Summaries are constrained by the model’s max token limit (128 tokens), which may omit important details for very long articles.

### Potential Improvements

1. Dataset Expansion: Incorporating datasets from diverse news domains to improve generalizability.
2. Model Optimization: Experimenting with larger T5 variants (e.g., T5-Large) for potentially better performance.

## Code Structure

### Files and Descriptions

1. News_Summarization.ipynb
Purpose: Covers the entire workflow from dataset loading, preprocessing, model fine-tuning, and evaluation.
Key Steps:
Loading the CNN/DailyMail dataset.
Preprocessing data: shuffling, sampling, and preparing inputs for T5-Base.
Fine-tuning the model using the Hugging Face Trainer API.
Evaluating the model with ROUGE scores and visualizing results.
2. gradio.ipynb
Purpose: Implements an interactive Gradio interface for generating summaries.
Features:
Input: Textbox for users to paste news articles.
Output: Generated summary, word count bar chart, and model performance visualizations.
3. Gradio Page.pdf
Content: Demonstrates the layout and functionality of the Gradio interface.

### Running the Code

1. Fine-Tuning:
Run [`News_Summarization.ipynb`](News_Summarization.ipynb) to fine-tune the T5-Base model.
2. Interactive Demo:
Execute [`gradio.ipynb`](gradio.ipynb) to launch the Gradio interface.

## Resource Links

1. Model Card: [google-t5/t5-base](https://huggingface.co/google-t5/t5-base)
2. Dataset Card: [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
3. Hugging Face Transformers: [Documentation](https://huggingface.co/docs/transformers/)
4. Gradio: [Documentation](https://gradio.app/)
5. Video Demonstration: []()

## References

1. [Raffel, Colin, et al. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.”](https://arxiv.org/abs/1910.10683)
2. [See, Abigail, et al. “Get To The Point: Summarization with Pointer-Generator Networks.”](https://arxiv.org/abs/1704.04368)
3. [Wolf, Thomas, et al. “Transformers: State-of-the-Art Natural Language Processing.”](https://arxiv.org/abs/1910.03771)
