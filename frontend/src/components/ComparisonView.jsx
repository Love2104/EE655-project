import AnalysisCard from "./AnalysisCard";

export default function ComparisonView({ result, currentTimes }) {
  return (
    <div className="space-y-5">
      <div className="rounded-[30px] border border-neon-lime/20 bg-gradient-to-r from-white/6 to-neon-lime/10 p-5 shadow-glow">
        <p className="text-xs uppercase tracking-[0.35em] text-neon-lime/90">Similarity Engine</p>
        <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h3 className="font-display text-4xl text-white">{result.similarity_score.toFixed(2)}%</h3>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">{result.comparison_summary}</p>
          </div>
          <div className="h-3 w-full max-w-sm overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-neon-amber via-neon-cyan to-neon-lime"
              style={{ width: `${result.similarity_score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <AnalysisCard analysis={result.video_a} currentTime={currentTimes.primary} heading="Video A" />
        <AnalysisCard analysis={result.video_b} currentTime={currentTimes.secondary} heading="Video B" />
      </div>
    </div>
  );
}
