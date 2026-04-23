import { useMemo, useState } from "react";
import { UploadPanel } from "./components/UploadPanel";
import { StatsBoard } from "./components/StatsBoard";
import { ImpactPanel } from "./components/ImpactPanel";
import { QueueChart } from "./components/QueueChart";
import { EventList } from "./components/EventList";
import { PlanTimeline } from "./components/PlanTimeline";
import { RiskBreakdown } from "./components/RiskBreakdown";
import { CycleChart } from "./components/CycleChart";
import { LogPanel } from "./components/LogPanel";
import { DirectionalLoad } from "./components/DirectionalLoad";
import "./index.css";

const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const withBase = (path) => {
  if (!API_BASE) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
};
const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState(null);

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
    setJobId(null);
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
      const submission = await response.json();
      if (!submission.job_id || !submission.status_url) {
        throw new Error("Сервер не вернул job_id для фоновой обработки");
      }
      setJobId(submission.job_id);

      let lastPayload = submission;
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const pollResponse = await fetch(withBase(submission.status_url));
        if (!pollResponse.ok) {
          throw new Error(`Не удалось получить статус задачи (${pollResponse.status})`);
        }
        lastPayload = await pollResponse.json();
        if (lastPayload.status === "completed") {
          setResult(lastPayload.result ?? null);
          setStatus("done");
          return;
        }
        if (lastPayload.status === "failed") {
          throw new Error(lastPayload.error || "Фоновая обработка завершилась ошибкой");
        }
        await sleep(1500);
      }
      throw new Error(
        `Задача ${submission.job_id} не завершилась вовремя. Последний статус: ${lastPayload.status}.`,
      );
    } catch (err) {
      console.error(err);
      setError(err.message || "Не удалось обработать видео");
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

  const queueHistory = result?.queue_history ?? [];
  const planHistory = result?.plan_history ?? [];
  const events = result?.events ?? [];
  const logs = result?.logs ?? [];
  const summary = result?.summary ?? null;
  const framesProcessed = result?.frames_processed ?? 0;

  return (
    <div className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Traffic Lab</p>
          <h1>Дэшборд контроля перекрёстка</h1>
          <p className="muted">
            Загружайте короткое видео, чтобы увидеть очереди, рискованные сцены и предложенный план фаз после фоновой обработки.
          </p>
          {jobId ? <p className="muted">Текущая задача: {jobId}</p> : null}
        </div>
        <div className={`status-chip ${status}`}>
          <span />
          {status === "processing" ? "Обработка" : status === "done" ? "Готово" : "Ожидание"}
        </div>
      </header>

      <section className="panel-grid">
        <UploadPanel
          fileName={file?.name ?? null}
          status={status}
          error={error}
          onFileChange={handleFileChange}
          onSubmit={handleSubmit}
          downloadLinks={downloadLinks}
        />
        <StatsBoard summary={summary} framesProcessed={framesProcessed} events={events} />
        <ImpactPanel queueHistory={queueHistory} events={events} planHistory={planHistory} />
      </section>

      <section className="data-grid">
        <QueueChart queueHistory={queueHistory} events={events} />
        <DirectionalLoad queueHistory={queueHistory} />
      </section>

      <section className="data-grid">
        <PlanTimeline planHistory={planHistory} />
        <LogPanel logs={logs} />
        <EventList events={events} />
      </section>

      <section className="data-grid">
        <RiskBreakdown events={events} />
        <CycleChart planHistory={planHistory} />
      </section>
    </div>
  );
}
