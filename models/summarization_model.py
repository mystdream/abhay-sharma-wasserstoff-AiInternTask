# models/summarization_model.py
from transformers import pipeline

class Summarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, text):
        try:
            summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Error generating summary: {str(e)}"
        return summary
