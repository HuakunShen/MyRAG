-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1024) -- 1536 works for OpenAI embeddings, change as needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (1024),
  collection_name text,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  source text,
  source_id bigint,
  collection text,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    source,
    source_id,
    collection,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where collection = collection_name and metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;





