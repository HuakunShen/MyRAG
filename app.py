import gradio as gr
import time
import random
import tempfile
from pathlib import Path
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from modules.utils import compute_sha512, get_all_collection_names, get_supabase, load_dotenv
# from modules.vectorstore import CustomSupabaseVectorStore, CustomDocument
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

supabase = get_supabase()
# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
supabase_db = SupabaseVectorStore(supabase, embeddings, "documents", "match_documents")

def upload_files2(collection_name: str | None, paths: list[Path]):
    if collection_name is None:
        raise Exception("No collection selected")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for path in paths:
        if path.suffix == ".txt":
            loader = TextLoader(str(path))
        elif path.suffix == ".html":
            loader = UnstructuredHTMLLoader(str(path))
        elif path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
        else:
            raise Exception("Unsupported file type")
        data = "\n".join([doc.page_content for doc in loader.load()])
        hash = compute_sha512(data)
        match_hash = supabase.table("sources").select("*").eq("sha512", hash).execute()
        if len(match_hash.data) > 0:
            raise Exception("File already exists")
        source = path.name
        inserted = supabase.table("sources").insert(
            {"sha512": hash, "collection": collection_name, "source": source}).execute()
        docs = loader.load_and_split(text_splitter)
        for doc in docs:
            doc.metadata["source"] = source
            doc.metadata["sha512"] = compute_sha512(doc.page_content)

        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )

def upload_files(collection_name: str | None, files: list[tempfile._TemporaryFileWrapper]):
    paths = [Path(file.name) for file in files]
    upload_files2(collection_name, paths)


def upload_directory(collection_name: str | None, directory: list[tempfile._TemporaryFileWrapper]):
    return directory


def delete_collection(name: str | None):
    if name is not None:
        supabase.table("collections").delete().eq('name', name).execute()
    return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")


def clear_collection(name: str | None):
    if name is not None:
        supabase.table("sources").delete().eq('collection', name).execute()
    return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")

with gr.Blocks() as demo:
    with gr.Tab("Collections"):
        collections_names = get_all_collection_names(supabase)
        new_collection_text_box = gr.Textbox(label="New Collection")
        submit_btn = gr.Button(value="Submit")
        collections_radios = gr.Radio(collections_names, label="Collections", info="All Collections")


        def new_collection(name: str, ):
            supabase.table("collections").insert({"name": name}).execute()
            return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")


        submit_btn.click(new_collection, inputs=new_collection_text_box, outputs=collections_radios)


        def refresh_collections():
            return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")


        with gr.Row():
            refresh_btn = gr.Button(value="Refresh")
            refresh_btn.click(refresh_collections, outputs=collections_radios)
            del_btn = gr.Button(value="Delete")
            del_btn.click(delete_collection, inputs=collections_radios, outputs=collections_radios)
    with gr.Tab("Input"):
        collections_names = get_all_collection_names(supabase)
        collections_radios = gr.Radio(collections_names, label="Collections", info="All Collections")
        with gr.Row():
            refresh_btn = gr.Button(value="Refresh")
            refresh_btn.click(refresh_collections, outputs=collections_radios)
            del_btn = gr.Button(value="Delete Collection")
            del_btn.click(delete_collection, inputs=collections_radios, outputs=collections_radios)
            clear_btn = gr.Button(value="Clear Collection")
            clear_btn.click(clear_collection, inputs=collections_radios, outputs=collections_radios)
        with gr.Row():
            with gr.Column():
                files = gr.File(label="Files", file_count="multiple")
                index_btn = gr.Button("Index Files")
                index_btn.click(upload_files, inputs=[collections_radios, files])
            with gr.Column():
                directory = gr.File(label="Directory", file_count="directory")
                index_btn = gr.Button("Index Directory")
                index_btn.click(upload_directory, inputs=[collections_radios, directory])
    with gr.Tab("Search DB"):
        query = gr.Textbox(label="Search Vector DB")
        submit_btn = gr.Button(value="Search")
        # textarea = gr.TextArea(label="Results", type="text", lines=10)
        result = gr.Markdown("Result")
        def search_db(query: str):
            result = supabase_db.similarity_search(query, 10)
            markdown = ""
            for doc in result:
                markdown += "## " + doc.metadata["source"] + "\n"
                markdown += str(doc.page_content) + "\n"
            return gr.Markdown(markdown)
        submit_btn.click(search_db, inputs=query, outputs=result)
    with gr.Tab("Ask AI"):
        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask Question")
        
        def respond(question: str, chat_history):
            found_docs = supabase_db.similarity_search(question, 1)
            doc = found_docs[0]
            doc.page_content = doc.page_content[:1000]
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=found_docs, question=question)
            chat_history.append((question, response))
            return "", chat_history

        query.submit(respond, [query, chatbot], [query, chatbot])
        with gr.Row():
            clear = gr.ClearButton([query, chatbot])
            submit_btn = gr.Button(value="Ask")
            submit_btn.click(respond, inputs=[query, chatbot], outputs=[query, chatbot])


if __name__ == "__main__":
    demo.launch()


