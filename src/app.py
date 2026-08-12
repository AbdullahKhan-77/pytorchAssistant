"""
app.py - Gradio web interface (the single entry point on Colab).
Wraps answer() from generate.py in a simple chat UI with example questions.
Run on Colab (after building the index):  python src/app.py
share=True gives a public link that works from any browser.
"""

import gradio as gr

from generate import answer


def chat_fn(question, history):
    if not question.strip():
        return "Please enter a question about PyTorch."
    response, _sources = answer(question)
    return response


demo = gr.ChatInterface(
    fn=chat_fn,
    title="PyTorch Documentation Assistant",
    description="Ask questions about PyTorch. Answers are grounded in the official "
                "documentation with source citations.",
    examples=[
        "How do I create a linear layer?",
        "How do I use the Adam optimizer?",
        "What optimizer should I use for training?",
        "How does autograd track gradients?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)