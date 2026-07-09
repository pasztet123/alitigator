create table if not exists public.eureka_interpretations (
    document_id text primary key,
    source text not null default 'eureka',
    source_type text not null default 'interpretation',
    index_name text not null,
    version_id bigint,
    template_id bigint,
    template_version_id bigint,
    category text,
    status text,
    subject text not null,
    signature text,
    author text,
    published_date timestamptz,
    published_at timestamptz,
    keywords jsonb not null default '[]'::jsonb,
    legal_provisions jsonb not null default '[]'::jsonb,
    issues jsonb not null default '[]'::jsonb,
    law_tags jsonb not null default '[]'::jsonb,
    query text not null default '',
    source_url text,
    content_html text not null default '',
    content_text text not null default '',
    content_text_clean text not null default '',
    content_sha256 text,
    attachments jsonb not null default '[]'::jsonb,
    raw_field_map jsonb not null default '{}'::jsonb,
    raw_search jsonb not null default '{}'::jsonb,
    raw_detail jsonb not null default '{}'::jsonb,
    retrieved_at timestamptz not null default now(),
    indexed_at timestamptz not null default now()
);

create table if not exists public.eureka_chunks (
    chunk_id text primary key,
    document_id text not null references public.eureka_interpretations(document_id) on delete cascade,
    chunk_index integer not null,
    chunk_text text not null,
    chunk_chars integer not null,
    signature text,
    published_date timestamptz,
    source_url text,
    subject text not null,
    category text,
    keywords_text text not null default '',
    legal_provisions_text text not null default '',
    issues_text text not null default '',
    law_tags_text text not null default '',
    embedding double precision[] not null default '{}',
    embedding_norm double precision not null default 0,
    embedding_model text not null default 'alitigator-hash-v1',
    search_vector tsvector generated always as (
        setweight(to_tsvector('simple', coalesce(subject, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(signature, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(keywords_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(legal_provisions_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(issues_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(law_tags_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(chunk_text, '')), 'C')
    ) stored
);

alter table if exists public.eureka_chunks add column if not exists embedding double precision[] not null default '{}';
alter table if exists public.eureka_chunks add column if not exists embedding_norm double precision not null default 0;
alter table if exists public.eureka_chunks add column if not exists embedding_model text not null default 'alitigator-hash-v1';

create index if not exists eureka_chunks_document_id_idx on public.eureka_chunks(document_id);
create index if not exists eureka_chunks_search_vector_idx on public.eureka_chunks using gin(search_vector);

create table if not exists public.legal_document_versions (
    document_id text not null,
    version_id text not null,
    document_type text not null,
    title text not null,
    citation text not null,
    jurisdiction text not null default 'PL',
    effective_from date not null,
    effective_to date,
    publication_date date,
    is_consolidated_text boolean not null default false,
    primary key (document_id, version_id)
);

create table if not exists public.legal_provisions (
    provision_id text primary key,
    document_id text not null,
    version_id text not null,
    citation text not null,
    article text not null,
    paragraph text,
    point text,
    letter text,
    provision_text text not null,
    effective_from date not null,
    effective_to date,
    status text not null check (status in ('active', 'repealed', 'unknown')),
    source_document_id text not null,
    source_chunk_ids jsonb not null default '[]'::jsonb,
    source_span_start integer not null default 0,
    source_span_end integer not null default 0,
    references jsonb not null default '[]'::jsonb,
    amends text,
    repealed_by text,
    tax_domain text not null default '',
    taxpayer_role text not null default '',
    legal_mechanism text not null default '',
    entailed_result_codes jsonb not null default '[]'::jsonb,
    foreign key (document_id, version_id)
        references public.legal_document_versions(document_id, version_id)
);

create index if not exists legal_provisions_exact_idx
    on public.legal_provisions(document_id, citation, effective_from, effective_to, status);

alter table if exists public.legal_provisions add column if not exists tax_domain text not null default '';
alter table if exists public.legal_provisions add column if not exists taxpayer_role text not null default '';
alter table if exists public.legal_provisions add column if not exists legal_mechanism text not null default '';
alter table if exists public.legal_provisions add column if not exists entailed_result_codes jsonb not null default '[]'::jsonb;

create or replace function public.array_dot_product(a double precision[], b double precision[])
returns double precision
language sql
immutable
as $$
    select coalesce(sum(a[idx] * b[idx]), 0.0)::double precision
    from generate_subscripts(a, 1) as idx
    where idx <= least(coalesce(array_length(a, 1), 0), coalesce(array_length(b, 1), 0));
$$;

create or replace function public.array_l2_norm(a double precision[])
returns double precision
language sql
immutable
as $$
    select sqrt(coalesce(sum(value * value), 0.0))::double precision
    from unnest(coalesce(a, '{}'::double precision[])) as value;
$$;

drop function if exists public.search_eureka_chunks(text, integer);

create or replace function public.search_eureka_chunks(
    search_query text,
    match_count integer default 6,
    query_embedding double precision[] default '{}'::double precision[],
    lexical_weight real default 0.65,
    semantic_weight real default 0.35
)
returns table (
    chunk_id text,
    document_id text,
    chunk_index integer,
    chunk_text text,
    subject text,
    signature text,
    published_date timestamptz,
    source_url text,
    category text,
    score real
)
language sql
stable
as $$
    with params as (
        select
            greatest(match_count, 1) as effective_limit,
            websearch_to_tsquery('simple', search_query) as ts_query,
            coalesce(query_embedding, '{}'::double precision[]) as query_embedding,
            public.array_l2_norm(coalesce(query_embedding, '{}'::double precision[])) as query_norm,
            greatest(lexical_weight, 0)::double precision as lexical_weight,
            greatest(semantic_weight, 0)::double precision as semantic_weight
    ),
    lexical_ranked as (
        select
            c.chunk_id,
            row_number() over (
                order by ts_rank_cd(c.search_vector, p.ts_query) desc, c.published_date desc nulls last, c.chunk_index asc
            ) as lexical_rank
        from public.eureka_chunks c
        cross join params p
        where c.search_vector @@ p.ts_query
        limit (select greatest(effective_limit * 20, 120) from params)
    ),
    semantic_ranked as (
        select
            c.chunk_id,
            row_number() over (
                order by (
                    public.array_dot_product(c.embedding, p.query_embedding)
                    / nullif(c.embedding_norm * p.query_norm, 0)
                ) desc nulls last,
                c.published_date desc nulls last,
                c.chunk_index asc
            ) as semantic_rank
        from public.eureka_chunks c
        cross join params p
        where p.query_norm > 0 and c.embedding_norm > 0 and cardinality(c.embedding) > 0
        limit (select greatest(effective_limit * 20, 120) from params)
    ),
    candidate_ids as (
        select chunk_id from lexical_ranked
        union
        select chunk_id from semantic_ranked
    )
    select
        c.chunk_id,
        c.document_id,
        c.chunk_index,
        c.chunk_text,
        c.subject,
        c.signature,
        c.published_date,
        c.source_url,
        c.category,
        (
            coalesce((select lexical_weight from params), 0) / (20 + coalesce(l.lexical_rank, 100000))
            + coalesce((select semantic_weight from params), 0) / (20 + coalesce(s.semantic_rank, 100000))
        )::real as score
    from candidate_ids ids
    join public.eureka_chunks c on c.chunk_id = ids.chunk_id
    left join lexical_ranked l on l.chunk_id = c.chunk_id
    left join semantic_ranked s on s.chunk_id = c.chunk_id
    order by score desc, published_date desc nulls last, chunk_index asc
    limit (select effective_limit from params);
$$;

create extension if not exists pgcrypto;

create table if not exists public.chat_threads (
    id uuid primary key default gen_random_uuid(),
    title text not null default 'Nowy wątek',
    archived boolean not null default false,
    last_message_preview text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.chat_messages (
    id uuid primary key default gen_random_uuid(),
    chat_id uuid not null references public.chat_threads(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    feedback_rating smallint check (feedback_rating between 1 and 5),
    feedback_comment text,
    feedback_created_at timestamptz,
    created_at timestamptz not null default now()
);

alter table if exists public.chat_messages add column if not exists feedback_rating smallint;
alter table if exists public.chat_messages add column if not exists feedback_comment text;
alter table if exists public.chat_messages add column if not exists feedback_created_at timestamptz;

alter table if exists public.chat_messages
    drop constraint if exists chat_messages_feedback_rating_check;
alter table if exists public.chat_messages
    add constraint chat_messages_feedback_rating_check
    check (feedback_rating is null or feedback_rating between 1 and 5);

create index if not exists chat_threads_updated_at_idx on public.chat_threads(updated_at desc);
create index if not exists chat_threads_archived_updated_at_idx on public.chat_threads(archived, updated_at desc);
create index if not exists chat_messages_chat_id_created_at_idx on public.chat_messages(chat_id, created_at asc);

create or replace function public.touch_chat_thread_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists chat_threads_set_updated_at on public.chat_threads;

create trigger chat_threads_set_updated_at
before update on public.chat_threads
for each row
execute function public.touch_chat_thread_updated_at();
