import { History } from "lucide-react";

export default function HistoryPanel({ items, onSelect }) {
  return (
    <div className="rounded-[30px] border border-white/10 bg-white/[0.045] p-5 shadow-glow">
      <div className="flex items-center gap-3">
        <div className="rounded-2xl border border-white/10 bg-white/8 p-3">
          <History className="h-5 w-5 text-neon-cyan" />
        </div>
        <div>
          <p className="font-display text-xl text-white">Recent sessions</p>
          <p className="text-sm text-slate-400">Stored locally in the browser for quick reference.</p>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        {items.length ? (
          items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item)}
              className="w-full rounded-[22px] border border-white/10 bg-pitch-900/80 px-4 py-4 text-left transition hover:border-white/25 hover:bg-white/8 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={!item.data}
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="font-medium text-white">{item.title}</p>
                  <p className="text-sm text-slate-400">{item.subtitle}</p>
                </div>
                <span className="text-xs uppercase tracking-[0.25em] text-neon-lime">
                  {item.data ? item.mode : `${item.mode} summary`}
                </span>
              </div>
            </button>
          ))
        ) : (
          <div className="rounded-[22px] border border-dashed border-white/15 px-4 py-8 text-center text-sm text-slate-400">
            Analyses will appear here after you run the model.
          </div>
        )}
      </div>
    </div>
  );
}
