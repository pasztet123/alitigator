create extension if not exists pgcrypto;

create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create table if not exists public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    email text,
    full_name text,
    law_firm text,
    stripe_customer_id text unique,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
before update on public.profiles
for each row
execute function public.touch_updated_at();

create or replace function public.handle_new_user_profile()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.profiles (id, email, full_name)
    values (
        new.id,
        new.email,
        coalesce(new.raw_user_meta_data ->> 'full_name', null)
    )
    on conflict (id) do update
    set email = excluded.email,
        full_name = coalesce(excluded.full_name, public.profiles.full_name);

    return new;
end;
$$;

drop trigger if exists on_auth_user_created_profile on auth.users;
create trigger on_auth_user_created_profile
after insert on auth.users
for each row
execute function public.handle_new_user_profile();

create table if not exists public.credit_ledger (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    amount integer not null check (amount <> 0),
    entry_type text not null check (entry_type in ('topup', 'usage', 'refund', 'adjustment')),
    source_type text not null,
    source_id text not null,
    description text not null default '',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (source_type, source_id)
);

create index if not exists credit_ledger_user_id_created_at_idx
on public.credit_ledger(user_id, created_at desc);

create or replace view public.user_token_balances as
select
    user_id,
    coalesce(sum(amount), 0)::integer as balance
from public.credit_ledger
group by user_id;

create table if not exists public.credit_orders (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    pack_id text not null,
    pack_name text not null,
    token_amount integer not null check (token_amount > 0),
    currency text not null default 'pln',
    unit_amount integer not null check (unit_amount > 0),
    stripe_customer_id text,
    stripe_checkout_session_id text unique,
    stripe_payment_intent_id text unique,
    checkout_url text,
    status text not null default 'pending' check (status in ('pending', 'paid', 'credited', 'failed', 'expired', 'refunded')),
    metadata jsonb not null default '{}'::jsonb,
    credited_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

drop trigger if exists credit_orders_set_updated_at on public.credit_orders;
create trigger credit_orders_set_updated_at
before update on public.credit_orders
for each row
execute function public.touch_updated_at();

alter table if exists public.chat_threads add column if not exists user_id uuid references auth.users(id) on delete cascade;
alter table if exists public.chat_threads add column if not exists archived boolean not null default false;
alter table if exists public.chat_threads add column if not exists last_message_preview text not null default '';
alter table if exists public.chat_threads add column if not exists created_at timestamptz not null default now();
alter table if exists public.chat_threads add column if not exists updated_at timestamptz not null default now();

create index if not exists chat_threads_user_id_updated_at_idx
on public.chat_threads(user_id, updated_at desc);

alter table if exists public.profiles enable row level security;
alter table if exists public.credit_ledger enable row level security;
alter table if exists public.credit_orders enable row level security;
alter table if exists public.chat_threads enable row level security;
alter table if exists public.chat_messages enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own" on public.profiles
for select using (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own" on public.profiles
for update using (auth.uid() = id);

drop policy if exists "credit_ledger_select_own" on public.credit_ledger;
create policy "credit_ledger_select_own" on public.credit_ledger
for select using (auth.uid() = user_id);

drop policy if exists "credit_orders_select_own" on public.credit_orders;
create policy "credit_orders_select_own" on public.credit_orders
for select using (auth.uid() = user_id);

drop policy if exists "chat_threads_select_own" on public.chat_threads;
create policy "chat_threads_select_own" on public.chat_threads
for select using (auth.uid() = user_id);

drop policy if exists "chat_threads_insert_own" on public.chat_threads;
create policy "chat_threads_insert_own" on public.chat_threads
for insert with check (auth.uid() = user_id);

drop policy if exists "chat_threads_update_own" on public.chat_threads;
create policy "chat_threads_update_own" on public.chat_threads
for update using (auth.uid() = user_id);

drop policy if exists "chat_messages_select_own" on public.chat_messages;
create policy "chat_messages_select_own" on public.chat_messages
for select using (
    exists (
        select 1
        from public.chat_threads threads
        where threads.id = chat_messages.chat_id
          and threads.user_id = auth.uid()
    )
);

drop policy if exists "chat_messages_insert_own" on public.chat_messages;
create policy "chat_messages_insert_own" on public.chat_messages
for insert with check (
    exists (
        select 1
        from public.chat_threads threads
        where threads.id = chat_messages.chat_id
          and threads.user_id = auth.uid()
    )
);
