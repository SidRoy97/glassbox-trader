// charting the cnn signal history per ticker
import { supabase, Decision } from "@/lib/supabase";
import { ActionBadge, Disclaimer } from "@/lib/ui";
import { ConfidenceLines } from "@/lib/charts";

export const dynamic = "force-dynamic";

export default async function Signals() {
  const { data } = await supabase.from("decisions")
    .select("decided_at,ticker,cnn_direction,cnn_confidence")
    .order("decided_at", { ascending: true }).limit(600);
  const rows = (data || []) as Pick<Decision,
    "decided_at" | "ticker" | "cnn_direction" | "cnn_confidence">[];

  // pivoting confidence per ticker into one date-indexed series
  const tickers = [...new Set(rows.map((r) => r.ticker))];
  const byDate: Record<string, Record<string, string | number>> = {};
  for (const r of rows) {
    const date = r.decided_at.slice(5, 10);
    byDate[date] ??= { date };
    byDate[date][r.ticker] = r.cnn_confidence;
  }
  const series = Object.values(byDate);

  // collecting the latest signal per ticker for the cards
  const latest: Record<string, typeof rows[number]> = {};
  for (const r of rows) latest[r.ticker] = r;

  return (
    <div>
      <h1 className="text-2xl font-bold">Signal engine</h1>
      <p className="text-sm text-zinc-500 mt-1">
        The CNN's direction call and confidence per ticker — the quantitative
        anchor every debate starts from.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
        {Object.values(latest).map((r) => (
          <div key={r.ticker} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <span className="font-bold">{r.ticker}</span>
              <ActionBadge action={r.cnn_direction} />
            </div>
            <div className="text-xs text-zinc-500 mt-2">
              confidence <span className="text-zinc-300 font-mono">
                {Math.round(r.cnn_confidence * 100)}%</span>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
        <h2 className="text-sm font-semibold text-zinc-300 mb-3">
          CNN confidence over time
        </h2>
        <ConfidenceLines data={series} tickers={tickers} />
      </div>

      <p className="text-xs text-zinc-600 mt-4">
        Confidence near 33% means the model sees no edge between Up, Down, and
        Neutral. The weekly review tracks this model's live hit rate and
        triggers retraining when it decays toward random.
      </p>
      <Disclaimer />
    </div>
  );
}
