import { Film, UploadCloud, Video } from "lucide-react";
import { useDropzone } from "react-dropzone";

function VideoDropzone({
  title,
  subtitle,
  file,
  previewUrl,
  onDrop,
  onClear,
  onTimeUpdate,
  accent = "cyan",
}) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "video/*": [".mp4", ".avi", ".mov", ".mkv"],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => onDrop(acceptedFiles[0] || null),
  });

  return (
    <div className="rounded-[28px] border border-white/10 bg-white/5 p-4 shadow-glow">
      <div className="mb-4 flex items-start justify-between">
        <div>
          <p className="font-display text-lg text-white">{title}</p>
          <p className="text-sm text-slate-400">{subtitle}</p>
        </div>
        <div
          className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.25em] ${
            accent === "lime"
              ? "border-neon-lime/40 text-neon-lime"
              : "border-neon-cyan/40 text-neon-cyan"
          }`}
        >
          {file ? "Loaded" : "Awaiting clip"}
        </div>
      </div>

      <div
        {...getRootProps()}
        className={`group flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-[24px] border border-dashed px-6 text-center transition ${
          isDragActive
            ? "border-neon-cyan bg-neon-cyan/10"
            : "border-white/15 bg-pitch-900/80 hover:border-white/35 hover:bg-white/6"
        }`}
      >
        <input {...getInputProps()} />
        {previewUrl ? (
          <div className="w-full">
            <video
              className="h-52 w-full rounded-[18px] object-cover shadow-2xl"
              controls
              src={previewUrl}
              onTimeUpdate={(event) => onTimeUpdate?.(event.currentTarget.currentTime)}
            />
            <div className="mt-4 flex items-center justify-between text-left">
              <div className="min-w-0">
                <p className="truncate font-medium text-white">{file?.name}</p>
                <p className="text-sm text-slate-400">Ready for inference</p>
              </div>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onClear();
                }}
                className="rounded-full border border-white/15 px-4 py-2 text-sm text-slate-200 transition hover:border-white/30 hover:bg-white/8"
              >
                Clear
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-4 rounded-full border border-white/10 bg-white/10 p-4">
              {title.includes("Primary") ? (
                <UploadCloud className="h-8 w-8 text-neon-cyan" />
              ) : (
                <Film className="h-8 w-8 text-neon-lime" />
              )}
            </div>
            <p className="font-display text-xl text-white">
              {isDragActive ? "Drop the cricket clip here" : "Drag and drop a shot video"}
            </p>
            <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
              Use MP4, AVI, MOV, or MKV. The backend will extract frames, classify the stroke,
              and generate timeline analytics.
            </p>
            <div className="mt-5 flex items-center gap-2 rounded-full border border-white/10 bg-white/6 px-4 py-2 text-sm text-slate-300">
              <Video className="h-4 w-4" />
              Click to browse files
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function UploadPanel({
  primaryFile,
  primaryPreview,
  secondaryFile,
  secondaryPreview,
  onPrimaryChange,
  onSecondaryChange,
  onPrimaryTimeUpdate,
  onSecondaryTimeUpdate,
  onAnalyze,
  loading,
}) {
  const compareMode = Boolean(secondaryFile);

  return (
    <section className="space-y-5">
      <div className="rounded-[30px] border border-white/10 bg-gradient-to-br from-pitch-900 via-pitch-800 to-pitch-900 p-6 shadow-glow">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm uppercase tracking-[0.35em] text-neon-cyan/80">Match Lab</p>
            <h2 className="mt-3 font-display text-3xl text-white">
              Upload one clip for classification or two for similarity analysis.
            </h2>
          </div>
          <button
            type="button"
            onClick={onAnalyze}
            disabled={!primaryFile || loading}
            className="rounded-full bg-gradient-to-r from-neon-cyan via-sky-400 to-neon-lime px-6 py-3 font-semibold text-slate-950 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Processing..." : compareMode ? "Run Comparison" : "Run Analysis"}
          </button>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <VideoDropzone
          title="Primary Video"
          subtitle="Required"
          file={primaryFile}
          previewUrl={primaryPreview}
          onDrop={onPrimaryChange}
          onClear={() => onPrimaryChange(null)}
          onTimeUpdate={onPrimaryTimeUpdate}
        />
        <VideoDropzone
          title="Secondary Video"
          subtitle="Optional for similarity"
          file={secondaryFile}
          previewUrl={secondaryPreview}
          onDrop={onSecondaryChange}
          onClear={() => onSecondaryChange(null)}
          onTimeUpdate={onSecondaryTimeUpdate}
          accent="lime"
        />
      </div>
    </section>
  );
}
