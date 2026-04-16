import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BrainCircuit, Flame, Radar, Timer } from "lucide-react";

const PIE_COLORS = ["#4df6ff", "#b7ff7a", "#ffb347", "#ff785a", "#38bdf8", "#f97316"];

function StatPill({ icon: Icon, label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/6 p-4">
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-black/20">
        <Icon className="h-5 w-5 text-neon-cyan" />
      </div>
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-1 font-display text-2xl text-white">{value}</p>
    </div>
  );
}

export default function AnalysisCard({ analysis, currentTime = 0, heading = "Analysis" }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  const livePoint = useMemo(() => {
    if (!analysis?.timeline?.length) {
      return null;
    }
    return analysis.timeline.reduce((closest, point) => {
      return Math.abs(point.time - currentTime) < Math.abs(closest.time - currentTime) ? point : closest;
    }, analysis.timeline[0]);
  }, [analysis, currentTime]);

  const timelineData = analysis.timeline.map((point) => ({
    time: `${point.time}s`,
    confidence: point.confidence,
    label: point.label,
  }));

  return (
    <div className="space-y-5 rounded-[30px] border border-white/10 bg-white/[0.045] p-5 shadow-glow">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-slate-400">{heading}</p>
          <h3 className="mt-2 font-display text-3xl text-white">{analysis.prediction.label}</h3>
          <p className="mt-2 text-sm text-slate-400">{analysis.metadata.filename}</p>
        </div>
        <div className="min-w-[220px] rounded-[22px] border border-neon-cyan/25 bg-neon-cyan/10 p-4">
          <div className="flex items-center justify-between text-sm text-slate-200">
            <span>Confidence</span>
            <span>{analysis.prediction.confidence.toFixed(2)}%</span>
          </div>
          <div className="mt-3 h-3 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-neon-cyan to-neon-lime"
              style={{ width: `${analysis.prediction.confidence}%` }}
            />
          </div>
          {livePoint ? (
            <p className="mt-3 text-xs uppercase tracking-[0.28em] text-neon-lime">
              Live readout: {livePoint.label} at {livePoint.confidence.toFixed(1)}%
            </p>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <StatPill icon={BrainCircuit} label="Top Class" value={analysis.prediction.label} />
        <StatPill
          icon={Timer}
          label="Duration"
          value={`${analysis.metadata.duration_seconds.toFixed(1)}s`}
        />
        <StatPill
          icon={Radar}
          label="Frames Sampled"
          value={`${analysis.metadata.sampled_frames}`}
        />
        <StatPill icon={Flame} label="FPS" value={analysis.metadata.fps.toFixed(1)} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <div className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-4">
          <div className="mb-4">
            <p className="font-display text-xl text-white">Frame-wise prediction timeline</p>
            <p className="text-sm text-slate-400">
              Sliding-window inference mapped across the clip to expose shifts in shot confidence.
            </p>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="confidenceFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#4df6ff" stopOpacity={0.9} />
                    <stop offset="95%" stopColor="#4df6ff" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148,163,184,0.15)" vertical={false} />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    background: "#07111f",
                    border: "1px solid rgba(148,163,184,0.2)",
                    borderRadius: "16px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="confidence"
                  stroke="#4df6ff"
                  fill="url(#confidenceFill)"
                  strokeWidth={3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-5">
          <div className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-4">
            <p className="font-display text-xl text-white">Shot probability mix</p>
            <div className="mt-4 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analysis.breakdown.slice(0, 6)}
                    dataKey="probability"
                    nameKey="label"
                    innerRadius={55}
                    outerRadius={90}
                    paddingAngle={4}
                  >
                    {analysis.breakdown.slice(0, 6).map((entry, index) => (
                      <Cell key={entry.label} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "#07111f",
                      border: "1px solid rgba(148,163,184,0.2)",
                      borderRadius: "16px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-4">
            <p className="font-display text-xl text-white">Top-3 predictions</p>
            <div className="mt-4 h-52">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analysis.top_predictions}>
                  <CartesianGrid stroke="rgba(148,163,184,0.15)" vertical={false} />
                  <XAxis dataKey="label" hide />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      background: "#07111f",
                      border: "1px solid rgba(148,163,184,0.2)",
                      borderRadius: "16px",
                    }}
                  />
                  <Bar dataKey="probability" radius={[10, 10, 0, 0]} fill="#b7ff7a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 space-y-2">
              {analysis.top_predictions.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/6 px-4 py-3 text-sm text-slate-200"
                >
                  <span>{item.label}</span>
                  <span>{item.probability.toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="font-display text-xl text-white">Key frames and visual explanation</p>
            <p className="text-sm text-slate-400">
              Toggle between raw frames and saliency overlays generated from model gradients.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowHeatmap((value) => !value)}
            className="rounded-full border border-white/15 px-4 py-2 text-sm text-slate-100 transition hover:border-white/30 hover:bg-white/8"
          >
            {showHeatmap ? "Show raw frames" : "Show heatmaps"}
          </button>
        </div>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {analysis.key_frames.map((frame) => (
            <div
              key={`${frame.time}-${frame.label}`}
              className="rounded-[22px] border border-white/10 bg-white/5 p-3"
            >
              <img
                alt={`${frame.label} key frame`}
                className="h-52 w-full rounded-[18px] object-cover"
                src={`data:image/jpeg;base64,${showHeatmap ? frame.heatmap_base64 : frame.image_base64}`}
              />
              <div className="mt-3 flex items-center justify-between text-sm text-slate-200">
                <span>{frame.label}</span>
                <span>{frame.time.toFixed(2)}s</span>
              </div>
              <p className="mt-1 text-xs uppercase tracking-[0.25em] text-neon-cyan">
                Confidence {frame.confidence.toFixed(2)}%
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-4">
        <p className="font-display text-xl text-white">Analyst notes</p>
        <div className="mt-4 space-y-3">
          {analysis.insights.map((insight) => (
            <div
              key={insight}
              className="rounded-2xl border border-white/10 bg-white/6 px-4 py-3 text-sm leading-6 text-slate-200"
            >
              {insight}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
