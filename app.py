import gradio as gr
import time
import random
import tempfile
from pathlib import Path
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from modules.utils import compute_sha512, get_all_collection_names, get_supabase, load_dotenv
from modules.vectorstore import CustomSupabaseVectorStore, CustomDocument
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

supabase = get_supabase()
# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
supabase_db = CustomSupabaseVectorStore(supabase, embeddings, "documents", "match_documents")


def upload_files_help(collection_name: str | None, paths: list[Path]):
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
        source_id = inserted.data[0].get('id')
        docs = loader.load_and_split(text_splitter)
        docs = [CustomDocument(page_content=doc.page_content, metadata=doc.metadata, source_id=source_id, source=source, collection=collection_name) for doc in docs]
        # for doc in docs:
        #     doc.metadata["collection"] = collection
        #     doc.metadata["source"] = source
        #     doc.metadata["sha512"] = compute_sha512(doc.page_content)
        db = CustomSupabaseVectorStore.from_docs(supabase, table_name="documents", documents=docs, embedding=embeddings, collection=collection_name)
        # vector_store = SupabaseVectorStore.from_documents(
        #     docs,
        #     embeddings,
        #     client=supabase,
        #     table_name="documents",
        #     query_name="match_documents",
        #     chunk_size=500,
        # )

def upload_files(collection_name: str | None, files: list[tempfile._TemporaryFileWrapper]):
    paths = [Path(file.name) for file in files]
    upload_files_help(collection_name, paths)


def upload_directory(collection_name: str | None, directory: list[tempfile._TemporaryFileWrapper]):
    print(directory)
    return directory

def refresh_collections():
    return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")


def delete_collection(name: str | None):
    if name is not None:
        supabase.table("collections").delete().eq('name', name).execute()
    return gr.Radio(get_all_collection_names(supabase), label="Collections", info="All Collections")

def show_warning(msg: str):
    return gr.Warning(msg)

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
        new_collection_text_box.submit(new_collection, inputs=[new_collection_text_box], outputs=[collections_radios])

        with gr.Row():
            refresh_btn = gr.Button(value="Refresh")
            refresh_btn.click(refresh_collections, outputs=collections_radios)
            del_btn = gr.Button(value="Delete")
            del_btn.click(delete_collection, inputs=collections_radios, outputs=collections_radios)
            clear_btn = gr.Button(value="Clear Collection")
            clear_btn.click(clear_collection, inputs=collections_radios, outputs=collections_radios)
        files = gr.File(label="Files", file_count="multiple")
        index_btn = gr.Button("Index Files")
        index_btn.click(upload_files, inputs=[collections_radios, files])
    # with gr.Tab("Input"):
    #     collections_names1 = get_all_collection_names(supabase)
    #     collections_radios1 = gr.Radio(collections_names1, label="Collections", info="All Collections")
    #     with gr.Row():
    #         refresh_btn1 = gr.Button(value="Refresh")
    #         refresh_btn1.click(refresh_collections, outputs=[collections_radios1])
    #         del_btn = gr.Button(value="Delete Collection")
    #         del_btn.click(delete_collection, inputs=collections_radios, outputs=collections_radios)
    #         clear_btn = gr.Button(value="Clear Collection")
    #         clear_btn.click(clear_collection, inputs=collections_radios, outputs=collections_radios)
    #     # with gr.Row():
    #     #     with gr.Column():
    #     files = gr.File(label="Files", file_count="multiple")
    #     index_btn = gr.Button("Index Files")
    #     index_btn.click(upload_files, inputs=[collections_radios, files])
            # with gr.Column():
            #     directory = gr.File(label="Directory", file_count="directory")
            #     index_btn = gr.Button("Index Directory")
            #     directory.change(upload_directory, inputs=[collections_radios, directory])
            #     index_btn.click(upload_directory, inputs=[collections_radios, directory])
    with gr.Tab("Search DB"):
        with gr.Row():
            with gr.Column():
                collections_names = get_all_collection_names(supabase)
                collections_radios = gr.Radio(collections_names, label="Collections", info="All Collections")
            with gr.Column():
                refresh_btn = gr.Button(value="Refresh")
                refresh_btn.click(refresh_collections, outputs=[collections_radios])  

        query = gr.Textbox(label="Search Vector DB")
        submit_btn = gr.Button(value="Search")
        result = gr.Markdown("Result")
        def search_db(collection: str | None, query: str):
            if collection is None:
                raise Exception("No collection selected")
            ret = supabase_db.similarity_search(query, collection, 10)
            markdown = ""
            for doc in ret:
                markdown += "## " + doc.source + "\n"
                markdown += str(doc.page_content) + "\n"
            return gr.Markdown(markdown)
        submit_btn.click(search_db, inputs=[collections_radios, query], outputs=result)
        query.submit(search_db, inputs=[collections_radios, query], outputs=[result])
    with gr.Tab("Ask AI"):
        with gr.Row():
            with gr.Column():
                collections_names = get_all_collection_names(supabase)
                collections_radios = gr.Radio(collections_names, label="Collections", info="All Collections")
            with gr.Column():
                refresh_btn = gr.Button(value="Refresh")
                refresh_btn.click(refresh_collections, outputs=[collections_radios])  

        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask Question")
        collections_radios.change(lambda x: "", outputs=[query])
        def respond(question: str, collections: str | None, chat_history):
            if collections is None:
                # return "", gr.Warning("No collection selected")
                # return "", chat_history
                raise ValueError("No collection selected")
                # return "", [ValueError("No collection selected")]
            found_docs = supabase_db.similarity_search(question, collections, 1)
            doc = found_docs[0]
            doc.page_content = doc.page_content[:1000]
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=found_docs, question=question)
            chat_history.append((question, response))
            return "", chat_history

        query.submit(respond, [query, collections_radios, chatbot], [query, chatbot])
        with gr.Row():
            clear = gr.ClearButton([query, chatbot])
            submit_btn = gr.Button(value="Ask")
            submit_btn.click(respond, inputs=[query, collections_radios, chatbot], outputs=[query, chatbot])


if __name__ == "__main__":
    demo.launch()


