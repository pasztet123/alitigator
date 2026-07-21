create table if not exists rag_documents (
    document_id varchar(191) primary key,
    content_sha256 varchar(64) null,
    source varchar(64) not null default 'eureka',
    source_type varchar(32) not null default 'interpretation',
    source_subtype varchar(32) not null default '',
    authority varchar(191) not null default '',
    jurisdiction varchar(16) not null default 'PL',
    act_title varchar(512) not null default '',
    publication varchar(255) not null default '',
    legal_state_date varchar(32) not null default '',
    source_pages_json longtext not null,
    subject text not null,
    signature varchar(255) null,
    published_date varchar(64) null,
    source_url text null,
    category varchar(255) null,
    keywords_json longtext not null,
    legal_provisions_json longtext not null,
    issues_json longtext not null,
    law_tags_json longtext not null,
    tax_domain varchar(64) not null default '',
    signature_family varchar(64) not null default '',
    question_text mediumtext not null,
    facts_text mediumtext not null,
    decision_text mediumtext not null,
    indexed_at varchar(64) not null,
    key idx_rag_documents_source_type (source_type),
    key idx_rag_documents_tax_domain (tax_domain),
    key idx_rag_documents_published_date (published_date),
    key idx_rag_documents_content_sha256 (content_sha256)
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;

create table if not exists rag_chunks (
    chunk_id varchar(191) primary key,
    document_id varchar(191) not null,
    chunk_index int not null,
    chunk_text mediumtext not null,
    chunk_chars int not null,
    provision_id varchar(255) not null default '',
    display_reference varchar(191) not null default '',
    search_text mediumtext not null,
    question_text mediumtext not null,
    facts_text mediumtext not null,
    tax_domain varchar(64) not null default '',
    fulltext key ft_rag_chunks_search (search_text, question_text, facts_text, tax_domain),
    key idx_rag_chunks_document_id (document_id),
    key idx_rag_chunks_document_chunk (document_id, chunk_index),
    key idx_rag_chunks_display_reference (display_reference),
    constraint fk_rag_chunks_document
        foreign key (document_id) references rag_documents(document_id)
        on delete cascade
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;

create table if not exists rag_chunks_citations (
    chunk_id varchar(191) not null,
    citation varchar(191) not null,
    primary key (chunk_id, citation),
    key idx_rag_chunk_citations_exact (citation, chunk_id),
    constraint fk_rag_chunk_citations_chunk
        foreign key (chunk_id) references rag_chunks(chunk_id)
        on delete cascade
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;

create table if not exists legal_document_versions (
    document_id varchar(191) not null,
    version_id varchar(191) not null,
    document_type varchar(32) not null,
    title varchar(512) not null,
    citation varchar(255) not null,
    jurisdiction varchar(16) not null default 'PL',
    effective_from date not null,
    effective_to date null,
    publication_date date null,
    is_consolidated_text boolean not null default false,
    primary key (document_id, version_id)
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;

create table if not exists legal_provisions (
    provision_id varchar(255) primary key,
    document_id varchar(191) not null,
    version_id varchar(191) not null,
    citation varchar(191) not null,
    article varchar(32) not null,
    paragraph varchar(32) null,
    point varchar(32) null,
    letter varchar(16) null,
    provision_text mediumtext not null,
    effective_from date not null,
    effective_to date null,
    status enum('active', 'repealed', 'unknown') not null,
    source_document_id varchar(191) not null,
    source_chunk_ids_json longtext not null,
    source_span_start int not null default 0,
    source_span_end int not null default 0,
    references_json longtext not null,
    amends varchar(255) null,
    repealed_by varchar(255) null,
    tax_domain varchar(64) not null default '',
    taxpayer_role varchar(64) not null default '',
    legal_mechanism varchar(128) not null default '',
    entailed_result_codes_json longtext not null default ('[]'),
    key idx_legal_provisions_exact (document_id, citation, effective_from, effective_to, status),
    constraint fk_legal_provisions_version foreign key (document_id, version_id)
        references legal_document_versions(document_id, version_id)
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;

alter table legal_provisions add column if not exists tax_domain varchar(64) not null default '';
alter table legal_provisions add column if not exists taxpayer_role varchar(64) not null default '';
alter table legal_provisions add column if not exists legal_mechanism varchar(128) not null default '';
alter table legal_provisions add column if not exists entailed_result_codes_json longtext not null default ('[]');

create table if not exists legal_document_cards (
    document_id varchar(191) not null,
    extractor_version varchar(96) not null,
    dictionary_version varchar(96) not null,
    content_sha256 varchar(64) not null,
    card_json longtext not null,
    created_at timestamp not null default current_timestamp,
    primary key (document_id, extractor_version, dictionary_version, content_sha256)
) engine=InnoDB default charset=utf8mb4 collate=utf8mb4_unicode_ci;
