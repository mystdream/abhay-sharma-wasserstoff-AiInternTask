from transformers import pipeline

class SummarizationModel:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_image(self, results, full_image_text):
        object_info = [f"{result['class']} (confidence: {result['score']:.2f})" for result in results]
        object_summary = ", ".join(object_info)

        summary_input = f"The image contains the following objects: {object_summary}. "
        if full_image_text:
            summary_input += f"The following text was extracted from the image: '{full_image_text}'. "
        summary_input += "Provide a brief summary of the image content."

        try:
            summary = self.summarizer(summary_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Error generating summary: {str(e)}"

        return summary
