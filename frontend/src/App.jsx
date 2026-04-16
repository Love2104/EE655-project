import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Download, LoaderCircle, Sparkles } from "lucide-react";
import AnalysisCard from "./components/AnalysisCard";
import ComparisonView from "./components/ComparisonView";
import HistoryPanel from "./components/HistoryPanel";
import UploadPanel from "./components/UploadPanel";
import { compareVideos, downloadReport, predictVideo } from "./lib/api";

const HISTORY_KEY = "cricket-shot-history-v1";

function useObjectUrl(file) {
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return url;
}

function buildHistoryItem(payload, mode) {
  if (mode === "compare") {
    return {
      id: payload.request_id,
      mode: "Compare",
      title: `${payload.video_a.prediction.label} vs ${payload.video_b.prediction.label}`,
      subtitle: `${payload.similarity_score.toFixed(2)}% similarity`,
      data: payload,
    };
  }

  return {
    id: payload.request_id,
    mode: "Single",
    title: payload.prediction.label,
    subtitle: `${payload.prediction.confidence.toFixed(2)}% confidence`,
    data: payload,
  };
}

function buildReportPayload(result, mode) {
  if (mode === "compare") {
    return {
      title: "Cricket Shot Similarity Report",
      generatedAt: new Date().toLocaleString(),
      summary: {
        label: `${result.video_a.prediction.label} vs ${result.video_b.prediction.label}`,
        confidence: result.similarity_score,
      },
      topPredictions: result.video_a.top_predictions,
      insights: [
        result.comparison_summary,
        ...result.video_a.insights.slice(0, 2),
        ...result.video_b.insights.slice(0, 2),
      ],
    };
  }

  return {
    title: "Cricket Shot Recognition Report",
    generatedAt: new Date().toLocaleString(),
    summary: result.prediction,
    topPredictions: result.top_predictions,
    insights: result.insights,
  };
}

export default function App() {
  const [primaryFile, setPrimaryFile] = useState(null);
  const [secondaryFile, setSecondaryFile] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playerTimes, setPlayerTimes] = useState({ primary: 0, secondary: 0 });
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  });

  const primaryPreview = useObjectUrl(primaryFile);
  const secondaryPreview = useObjectUrl(secondaryFile);

  function handlePrimaryChange(file) {
    setPrimaryFile(file);
    setPlayerTimes((current) => ({ ...current, primary: 0 }));
  }

  function handleSecondaryChange(file) {
    setSecondaryFile(file);
    setPlayerTimes((current) => ({ ...current, secondary: 0 }));
  }

  useEffect(() => {
    const persisted = history.slice(0, 6).map(({ data, ...rest }) => rest);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(persisted));
  }, [history]);

  async function runAnalysis() {
    if (!primaryFile) {
      setError("Upload at least one cricket shot video to begin.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      if (secondaryFile) {
        const payload = await compareVideos(primaryFile, secondaryFile);
        setCompareResult(payload);
        setSingleResult(null);
        setHistory((prev) => [buildHistoryItem(payload, "compare"), ...prev].slice(0, 6));
      } else {
        const payload = await predictVideo(primaryFile);
        setSingleResult(payload);
        setCompareResult(null);
        setHistory((prev) => [buildHistoryItem(payload, "single"), ...prev].slice(0, 6));
      }
    } catch (caughtError) {
      setError(caughtError.message || "Something went wrong while running inference.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDownloadReport() {
    const result = compareResult || singleResult;
    if (!result) {
      return;
    }

    const reportMode = compareResult ? "compare" : "single";
    const blob = await downloadReport(buildReportPayload(result, reportMode));
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "cricket-shot-report.pdf";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function handleHistorySelect(item) {
    if (!item.data) {
      setError("Detailed history is available for the current browser session after analysis runs.");
      return;
    }

    if (item.mode === "Compare") {
      setCompareResult(item.data);
      setSingleResult(null);
      return;
    }

    setSingleResult(item.data);
    setCompareResult(null);
  }

  return (
    <div className="min-h-screen bg-pitch-950 bg-pitch-grid bg-[size:22px_22px] text-slate-100">
      <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
        <header className="relative overflow-hidden rounded-[36px] border border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(77,246,255,0.22),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(183,255,122,0.18),_transparent_28%),linear-gradient(135deg,_rgba(7,17,31,0.96),_rgba(19,34,56,0.95))] p-6 shadow-glow sm:p-8">
          <div className="absolute right-6 top-6 h-36 w-36 rounded-full bg-neon-cyan/10 blur-3xl" />
          <div className="absolute bottom-4 left-1/3 h-28 w-28 rounded-full bg-neon-lime/10 blur-3xl" />
          <div className="relative flex flex-wrap items-end justify-between gap-5">
            <div className="max-w-4xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.35em] text-neon-cyan">
                <Sparkles className="h-4 w-4" />
                Cricket Analytics Studio
              </div>
              <h1 className="mt-5 font-display text-4xl leading-tight text-white sm:text-6xl">
                Recognize cricket shots, inspect frame-level behavior, and compare batting mechanics.
              </h1>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300 sm:text-lg">
                Built around the existing EfficientNetB0 + GRU model weights, with a modern dashboard
                for upload, similarity analysis, key-frame explanations, and downloadable reports.
              </p>
            </div>
            <div className="grid gap-3 rounded-[28px] border border-white/10 bg-black/20 p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/10 p-3">
                  <BarChart3 className="h-5 w-5 text-neon-cyan" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Outputs</p>
                  <p className="font-display text-xl text-white">Prediction, timeline, similarity</p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={!singleResult && !compareResult}
                className="inline-flex items-center justify-center gap-2 rounded-full border border-white/15 px-5 py-3 text-sm font-semibold text-white transition hover:border-white/30 hover:bg-white/8 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Download className="h-4 w-4" />
                Download PDF report
              </button>
            </div>
          </div>
        </header>

        <main className="mt-6 grid gap-6 xl:grid-cols-[1.02fr_1.28fr]">
          <div className="space-y-6">
            <UploadPanel
              primaryFile={primaryFile}
              primaryPreview={primaryPreview}
              secondaryFile={secondaryFile}
              secondaryPreview={secondaryPreview}
              onPrimaryChange={handlePrimaryChange}
              onSecondaryChange={handleSecondaryChange}
              onPrimaryTimeUpdate={(time) =>
                setPlayerTimes((current) => ({ ...current, primary: time }))
              }
              onSecondaryTimeUpdate={(time) =>
                setPlayerTimes((current) => ({ ...current, secondary: time }))
              }
              onAnalyze={runAnalysis}
              loading={loading}
            />
            <HistoryPanel items={history} onSelect={handleHistorySelect} />
          </div>

          <section className="space-y-6">
            {error ? (
              <div className="rounded-[24px] border border-red-500/30 bg-red-500/10 px-5 py-4 text-sm text-red-100">
                {error}
              </div>
            ) : null}

            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  className="flex min-h-[500px] flex-col items-center justify-center rounded-[30px] border border-white/10 bg-white/[0.045] p-10 shadow-glow"
                >
                  <LoaderCircle className="h-14 w-14 animate-spin text-neon-cyan" />
                  <p className="mt-6 font-display text-3xl text-white">Running shot intelligence</p>
                  <p className="mt-3 max-w-lg text-center text-sm leading-6 text-slate-400">
                    Extracting frames, scoring the sequence, generating timeline windows, and building
                    visual explanations from the trained model.
                  </p>
                </motion.div>
              ) : compareResult ? (
                <motion.div
                  key="compare"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                >
                  <ComparisonView result={compareResult} currentTimes={playerTimes} />
                </motion.div>
              ) : singleResult ? (
                <motion.div
                  key="single"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                >
                  <AnalysisCard
                    analysis={singleResult}
                    currentTime={playerTimes.primary}
                    heading="Recognition Result"
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -12 }}
                  className="flex min-h-[500px] flex-col justify-between rounded-[30px] border border-white/10 bg-white/[0.045] p-6 shadow-glow"
                >
                  <div>
                    <p className="text-xs uppercase tracking-[0.35em] text-neon-cyan/80">Results Deck</p>
                    <h2 className="mt-4 font-display text-4xl text-white">Analytics will appear here.</h2>
                    <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
                      Upload a cricket shot video to see the predicted shot type, confidence flow,
                      key frames, probability breakdown, and visual explanation. Add a second video to
                      unlock similarity scoring and side-by-side comparison.
                    </p>
                  </div>
                  <div className="grid gap-4 md:grid-cols-3">
                    {[
                      "Top-3 predictions with confidence bars",
                      "Frame timeline analysis for changing certainty",
                      "Key frame heatmaps and downloadable reports",
                    ].map((copy) => (
                      <div
                        key={copy}
                        className="rounded-[24px] border border-white/10 bg-pitch-900/80 p-5 text-sm leading-6 text-slate-300"
                      >
                        {copy}
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </main>
      </div>
    </div>
  );
}
