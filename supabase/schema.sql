-- Vigil observability backend — run once in the Supabase SQL Editor
-- (Dashboard → SQL Editor → paste → Run). Creates the single event stream that
-- the backend writes to and the judge-facing Vercel view reads from.

create table if not exists public.vigil_events (
  id          bigint generated always as identity primary key,
  created_at  timestamptz not null default now(),
  type        text not null,          -- perception | fused | decision | call_status | escalation | note | agent_input | tool_call | conversation_turn
  source      text,                   -- vision | audio | claude | elevenlabs | nurse | agent | vigil
  patient     text,                   -- patient name / id
  summary     text,                   -- human-readable one-liner for the live feed
  payload     jsonb not null default '{}'::jsonb
);

create index if not exists vigil_events_created_idx on public.vigil_events (created_at desc);
create index if not exists vigil_events_type_idx on public.vigil_events (type);

-- RLS: the SECRET key (server) bypasses RLS to insert; the PUBLISHABLE key (frontend)
-- may read the stream so judges can watch it live.
alter table public.vigil_events enable row level security;

drop policy if exists "public read vigil_events" on public.vigil_events;
create policy "public read vigil_events"
  on public.vigil_events for select
  to anon, authenticated
  using (true);

-- Runtime rendezvous: the Mac backend publishes its current public tunnel URL here
-- (id='backend') so the public Vercel page discovers it at load time and survives
-- tunnel restarts with no redeploy. Server upserts with the SECRET key; the
-- PUBLISHABLE key may read it.
create table if not exists public.vigil_runtime (
  id          text primary key,
  url         text not null,
  updated_at  timestamptz not null default now()
);

alter table public.vigil_runtime enable row level security;

drop policy if exists "public read vigil_runtime" on public.vigil_runtime;
create policy "public read vigil_runtime"
  on public.vigil_runtime for select
  to anon, authenticated
  using (true);

-- Stream new rows to the frontend over Supabase Realtime.
do $$ begin alter publication supabase_realtime add table public.vigil_events; exception when duplicate_object then null; end $$;
do $$ begin alter publication supabase_realtime add table public.vigil_runtime; exception when duplicate_object then null; end $$;
