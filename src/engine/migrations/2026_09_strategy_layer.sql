-- strategy layer migration — paste into Supabase: SQL Editor -> New query -> Run
-- additive only: old cnn_* columns stay nullable so history remains readable

alter table decisions add column if not exists strategy_name text;
alter table decisions add column if not exists strategy_direction text;
alter table decisions add column if not exists strategy_score real;

create table if not exists strategy_backtests (
    run_date date not null,
    strategy text not null,
    utility real,
    sharpe real,
    cagr real,
    max_drawdown real,
    hit_rate real,
    exposure real,
    avg_turnover real,
    n_days integer,
    universe_size integer,
    created_at timestamptz not null default now(),
    primary key (run_date, strategy)
);

alter table strategy_backtests enable row level security;
create policy "public read strategy_backtests" on strategy_backtests
    for select using (true);
