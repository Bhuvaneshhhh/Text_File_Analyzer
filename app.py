import gradio as gr
import google.generativeai as genai
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set. Please set it in the Hugging Face Space secrets.")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

def analyze_input(text, file):
    try:
        if file is not None:
            text = file.decode("utf-8")  
        elif not text.strip():
            return "Error: Please enter text or upload a file.", ""
       
        text = text[:2000]  # Limit input size
        prompt = f"Analyze and summarize this document:\n\n{text}"
        response = model.generate_content([prompt], stream=True)  

        # Collect streamed response
        result = "".join([chunk.text for chunk in response])
        word_count = len(text.split())

        return result, f"Word Count: {word_count}"
    except Exception as e:
        return f"Error: {str(e)}", ""

def clear_inputs():
    return "", None, "", ""

def generate_downloadable_file(text):
    if text.strip():
        return gr.File.update(value=("analysis_result.txt", text.encode("utf-8")), visible=True)
    return gr.File.update(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("## AI-Powered Text & File Analyzer\nUpload a `.txt` file or enter text manually.")

    with gr.Row():
        text_input = gr.Textbox(label="Enter Text", placeholder="Type or paste your text here...", lines=6)
        file_input = gr.File(label="Upload Text File (.txt)", type="binary")

    output_text = gr.Textbox(label="Analysis Result", lines=10, interactive=False)
    word_count_display = gr.Textbox(label="Word Count", interactive=False)

    with gr.Row():
        analyze_button = gr.Button("Analyze", variant="primary")
        clear_button = gr.Button("Clear", variant="secondary")

    with gr.Column():
        gr.Markdown("### Download Analysis Result")
        with gr.Row():
            download_button = gr.Button("Download Result", variant="success", size="sm")
            download_file = gr.File(label="Click to Download", interactive=False, visible=False)

    analyze_button.click(analyze_input, inputs=[text_input, file_input], outputs=[output_text, word_count_display])
    clear_button.click(clear_inputs, inputs=[], outputs=[text_input, file_input, output_text, word_count_display])
    download_button.click(generate_downloadable_file, inputs=output_text, outputs=download_file)

demo.launch()
