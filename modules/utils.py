import os
import hashlib
from supabase.client import Client, create_client
from pathlib2 import Path

def compute_sha512(input_string):
    sha512_hash = hashlib.sha512()
    sha512_hash.update(input_string.encode('utf-8'))
    return sha512_hash.hexdigest()

def get_supabase() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase

def get_all_collection_names(supabase: Client) -> list[str]:
    collections = supabase.table("collections").select("*").execute()
    collections_data = collections.data
    collections_names = [collection["name"] for collection in collections_data]
    return collections_names

def load_dotenv(dotenv_path: Path | None=None):
    if dotenv_path is None:
        dotenv_path = Path(os.getcwd()) / ".env"
    with open(dotenv_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            key, value = line.split("=")
            os.environ[key] = value

