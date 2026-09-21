-- walk-forward + benchmark columns for strategy_backtests
-- paste into Supabase SQL Editor -> New query -> Run. additive, safe to re-run.

alter table strategy_backtests add column if not exists worst_fold_utility real;
alter table strategy_backtests add column if not exists mean_fold_utility real;
alter table strategy_backtests add column if not exists fold_spread real;
alter table strategy_backtests add column if not exists n_folds integer;
alter table strategy_backtests add column if not exists benchmark boolean default false;
