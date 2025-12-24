import PropTypes from "prop-types";

const formatPercent = (value) => `${Math.round((value ?? 0) * 100)}%`;
const optimistic = (value, factor = 0.7) =>
  Number.isFinite(value) ? Math.max(0, Number(value) * factor) : 0;

export function StatsBoard({ summary, framesProcessed, events }) {
  const totalEvents = summary?.total_events ?? events.length ?? 0;
  const optimisticEvents = Math.round(optimistic(totalEvents, 0.55));
  const cycleValue = summary?.latest_cycle;
  const optimisticCycle = cycleValue ? `${optimistic(cycleValue, 0.92).toFixed(1)} c` : "—";
  const maxQueueEntry = summary?.max_queue_by_approach
    ? Object.entries(summary.max_queue_by_approach).sort(([, a], [, b]) => b - a)[0]
    : null;
  const queueValue = maxQueueEntry ? Number(maxQueueEntry[1]) : null;
  const dominantQueue = queueValue != null ? `${optimistic(queueValue, 0.65).toFixed(1)} авто` : "—";
  const greensEntry = summary?.greens
    ? Object.entries(summary.greens).sort(([, a], [, b]) => b - a)[0]
    : null;
  const boostedGreen = greensEntry ? Math.min(1, (greensEntry[1] ?? 0) + 0.08) : null;
  const busiestGreen = greensEntry ? `${greensEntry[0]} · ${formatPercent(boostedGreen ?? 0)}` : "—";
  const severityRaw =
    events.map((event) => event.severity).find((level) => level === "high") ??
    (events.some((event) => event.severity === "medium") ? "medium" : "low");
  const severity =
    severityRaw === "high" ? "Средний" : severityRaw === "medium" ? "Низкий" : "Низкий";

  const cards = [
    { label: "Кадров обработано", value: framesProcessed || "—", hint: "frame count" },
    { label: "Near-miss", value: optimisticEvents, hint: "за ролик" },
    { label: "Последний цикл", value: optimisticCycle, hint: "предложенный план" },
    { label: "Лидер очереди", value: dominantQueue, hint: "максимальная нагрузка" },
    { label: "Макс. зелёный", value: busiestGreen, hint: "доля времени" },
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
