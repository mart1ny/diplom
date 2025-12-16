import { useMemo, useState } from "react";
import { HistorySection } from "./components/HistorySection";
import { SummaryPanel } from "./components/SummaryPanel";
import { UploadZone } from "./components/UploadZone";
import "./index.css";

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const withBase = (path) => {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
};

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files?.[0] ?? null);
    setError("");
  };

  const handleSubmit = async () => {
    if (!file || status === "processing") {
      return;
    }
    setStatus("processing");
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(withBase("/api/process-video"), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Сервер вернул статус ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      console.error(err);
      setError(err.message || "Не удалось загрузить видео");
      setStatus("idle");
    }
  };

  const downloadLinks = useMemo(() => {
    if (!result) {
      return [];
    }
    const links = [];
    if (result.output_video_url) {
      links.push({ label: "Аннотированное видео", href: withBase(result.output_video_url) });
    }
    if (result.events_file_url) {
      links.push({ label: "JSONL с событиями", href: withBase(result.events_file_url) });
    }
    return links;
  }, [result]);

  return (
    <div className="app-shell">
      <header style={{ marginBottom: "1.5rem" }}>
        <p className="hero-subtitle">CV + Near-miss + Оптимизация фаз</p>
        <h1 className="hero-title">Traffic Intelligence Dashboard</h1>
        <p className="hero-subtitle">
          Загрузите видео перекрёстка — пайплайн посчитает очереди, near-miss события и предложит план фаз.
        </p>
      </header>

      <UploadZone
        onFileChange={handleFileChange}
        onSubmit={handleSubmit}
        fileName={file?.name ?? null}
        disabled={status === "processing"}
        status={status}
      />

      {error && (
        <div className="card" style={{ marginTop: "1rem", color: "#b91c1c" }}>
          <strong>Ошибка:</strong> {error}
        </div>
      )}

      <div style={{ marginTop: "1.25rem" }}>
        <SummaryPanel summary={result?.summary} duration={result?.frames_processed ?? 0} />
      </div>

      {downloadLinks.length > 0 && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <h3>Результаты</h3>
          <div className="link-list">
            {downloadLinks.map((link) => (
              <a href={link.href} key={link.label} target="_blank" rel="noreferrer">
                ⬇ {link.label}
              </a>
            ))}
          </div>
        </div>
      )}

      <HistorySection queueHistory={result?.queue_history ?? []} planHistory={result?.plan_history ?? []} />
    </div>
  );
}
