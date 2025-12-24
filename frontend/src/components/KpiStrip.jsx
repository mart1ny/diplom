import PropTypes from "prop-types";

const formatNumber = (value) => (value ?? 0).toLocaleString("ru-RU");

export function KpiStrip({ summary, framesProcessed, queueHistory }) {
  const nearMiss = summary?.total_events ?? 0;
  const nearMissRate = framesProcessed > 0 ? ((nearMiss / framesProcessed) * 1000).toFixed(1) : null;
  const avgQueuePeak = queueHistory?.length
    ? queueHistory.reduce((sum, entry) => sum + Math.max(0, ...Object.values(entry.queues ?? {})), 0) /
      queueHistory.length
    : null;
  const dominantRisk = summary?.risk_peaks
    ? Object.entries(summary.risk_peaks).sort(([, a], [, b]) => b - a)[0]
    : null;

  const kpis = [
    {
      label: "Near-miss",
      value: nearMiss,
      hint: "событий за ролик",
    },
    {
      label: "Rate",
      value: nearMissRate ?? "—",
      hint: nearMissRate ? "на 1000 кадров" : "нет данных",
    },
    {
      label: "Кадры",
      value: framesProcessed ?? 0,
      hint: "аннотировано",
    },
    {
      label: "Цикл",
      value: summary?.latest_cycle ? `${summary.latest_cycle.toFixed(1)} c` : "—",
      hint: "последняя оптимизация",
    },
    {
      label: "Очередь",
      value: avgQueuePeak ? `${avgQueuePeak.toFixed(1)} авто` : "—",
      hint: "средний пик",
    },
    dominantRisk && {
      label: "Риск",
      value: `${dominantRisk[0]} · ${(dominantRisk[1] * 100).toFixed(0)}%`,
      hint: "лидер по near-miss",
    },
  ].filter(Boolean);

  return (
    <div className="kpi-strip">
      {kpis.map((kpi) => (
        <div key={kpi.label} className="kpi-item">
          <p className="kpi-label">{kpi.label}</p>
          <strong>{typeof kpi.value === "number" ? formatNumber(kpi.value) : kpi.value}</strong>
          <span>{kpi.hint}</span>
        </div>
      ))}
    </div>
  );
}

KpiStrip.propTypes = {
  summary: PropTypes.shape({
    total_events: PropTypes.number,
    latest_cycle: PropTypes.number,
    risk_peaks: PropTypes.object,
  }),
  framesProcessed: PropTypes.number,
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      queues: PropTypes.object,
    }),
  ),
};

KpiStrip.defaultProps = {
  summary: null,
  framesProcessed: 0,
  queueHistory: [],
};
