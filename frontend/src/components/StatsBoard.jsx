import PropTypes from "prop-types";

const formatPercent = (value) => `${Math.round((value ?? 0) * 100)}%`;
const formatNumber = (value, digits = 1) =>
  Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
const formatSolverStatus = (value) => {
  if (!value) {
    return "—";
  }
  return String(value).replaceAll("_", " ");
};

export function StatsBoard({ summary, framesProcessed, events }) {
  const totalEvents = summary?.total_events ?? events.length ?? 0;
  const cycleValue = summary?.latest_cycle;
  const maxQueueEntry = summary?.max_queue_by_approach
    ? Object.entries(summary.max_queue_by_approach).sort(([, a], [, b]) => b - a)[0]
    : null;
  const queueValue = maxQueueEntry ? Number(maxQueueEntry[1]) : null;
  const greensEntry = summary?.greens
    ? Object.entries(summary.greens).sort(([, a], [, b]) => b - a)[0]
    : null;
  const durationsEntry = summary?.durations
    ? Object.entries(summary.durations).sort(([, a], [, b]) => b - a)[0]
    : null;
  const severityRaw =
    events.map((event) => event.severity).find((level) => level === "high") ??
    (events.some((event) => event.severity === "medium") ? "medium" : "low");
  const severity =
    severityRaw === "high" ? "Высокий" : severityRaw === "medium" ? "Средний" : "Низкий";

  const cards = [
    { label: "Кадров обработано", value: framesProcessed || "—", hint: "frame count" },
    { label: "Near-miss", value: totalEvents, hint: "за ролик" },
    { label: "Последний цикл", value: cycleValue ? `${formatNumber(cycleValue)} c` : "—", hint: "решение LP" },
    {
      label: "Лидер очереди",
      value: maxQueueEntry ? `${maxQueueEntry[0]} · ${formatNumber(queueValue)} авто` : "—",
      hint: "максимальная нагрузка",
    },
    {
      label: "Длинная фаза",
      value: durationsEntry ? `${durationsEntry[0]} · ${formatNumber(durationsEntry[1])} c` : "—",
      hint: "последний план",
    },
    {
      label: "LP solver",
      value: summary?.optimizer ? `${summary.optimizer} · ${formatSolverStatus(summary.solver_status)}` : "—",
      hint: summary?.objective_value != null ? `obj=${formatNumber(summary.objective_value, 2)}` : "решатель",
    },
    { label: "Уровень риска", value: severity, hint: "по свежим событиям" },
  ];

  return (
    <div className="card stats-board">
      {cards.map((card) => (
        <div key={card.label} className="stat-item">
          <p className="muted tiny">{card.label}</p>
          <strong>{card.value}</strong>
          <span>{card.hint}</span>
        </div>
      ))}
    </div>
  );
}

StatsBoard.propTypes = {
  summary: PropTypes.shape({
    total_events: PropTypes.number,
    latest_cycle: PropTypes.number,
    max_queue_by_approach: PropTypes.object,
    greens: PropTypes.object,
    durations: PropTypes.object,
    optimizer: PropTypes.string,
    solver_status: PropTypes.string,
    objective_value: PropTypes.number,
  }),
  framesProcessed: PropTypes.number,
  events: PropTypes.arrayOf(
    PropTypes.shape({
      severity: PropTypes.string,
    }),
  ),
};

StatsBoard.defaultProps = {
  summary: null,
  framesProcessed: 0,
  events: [],
};
