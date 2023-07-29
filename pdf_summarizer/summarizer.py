from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
import gradio as gr

llm = OpenAI(temperature=0)

def summarize_pdf(pdf_file):
    pdf_file_path = pdf_file.name
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

def main():
    input_pdf = gr.File(label="PDF file")
    output_summary = gr.outputs.Textbox(label="Summary")

    iface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf,
        outputs=output_summary,
        title="PDF Summarizer",
        description="Enter the path to a PDF file and get its summary.",
    )

    iface.launch()

if __name__ == "__main__":
    main()
