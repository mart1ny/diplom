import PropTypes from "prop-types";

const formatPercent = (value) => `${Math.round((value ?? 0) * 100)}%`;

export function InsightsPanel({ summary, selectedSignal, events }) {
  const maxQueueEntry = summary
    ? Object.entries(summary.max_queue_by_approach ?? {}).sort(([, a], [, b]) => b - a)[0]
    : null;
  const riskEntry = summary
    ? Object.entries(summary.risk_peaks ?? {}).sort(([, a], [, b]) => b - a)[0]
    : null;
  const greensEntry = summary
    ? Object.entries(summary.greens ?? {}).sort(([, a], [, b]) => b - a)[0]
    : null;
  const severeEvent = events?.length
    ? events.reduce((max, event) => ((event?.risk_score ?? 0) > (max?.risk_score ?? 0) ? event : max), events[0])
    : null;

  const insights = [
    {
      title: "Критический подход",
      metric: maxQueueEntry ? `${maxQueueEntry[0]} · ${maxQueueEntry[1]} авто` : "—",
      body: "Максимальная очередь требует продления зелёной фазы.",
    },
    {
      title: "Пик риска",
      metric: riskEntry ? `${riskEntry[0]} · ${formatPercent(riskEntry[1])}` : "—",
      body: "Near-miss события концентрации требуют внимания операторов.",
    },
    {
      title: "Эффективность зелёного",
      metric: greensEntry ? `${greensEntry[0]} · ${formatPercent(greensEntry[1])}` : "—",
      body: "Подход с наибольшей долей зелёного может быть донором времени.",
    },
    {
      title: "Свежий near-miss",
      metric: severeEvent ? `Кадр ${severeEvent.frame} · риск ${formatPercent(severeEvent.risk_score)}` : "—",
      body: severeEvent
        ? `Треки ${severeEvent.id1} и ${severeEvent.id2} — требуется проверка траекторий.`
        : "В текущем ролике рискованные события не обнаружены.",
    },
  ];

  return (
    <div className="card insights-card">
      <div className="panel-header">
        <p className="eyebrow">Инсайты</p>
        <h3>Рекомендации по сценарию</h3>
        <p className="muted">Собраны автоматически для выбранного светофора ({selectedSignal}).</p>
      </div>
      <div className="insights-grid">
        {insights.map((insight, index) => (
          <div key={insight.title} className="insight">
            <span className="chip ghost">#{index + 1}</span>
            <p className="insight-title">{insight.title}</p>
            <strong>{insight.metric}</strong>
            <p className="muted">{insight.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

InsightsPanel.propTypes = {
  summary: PropTypes.shape({
    max_queue_by_approach: PropTypes.object,
    risk_peaks: PropTypes.object,
    greens: PropTypes.object,
  }),
  selectedSignal: PropTypes.string.isRequired,
  events: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      risk_score: PropTypes.number,
      id1: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
      id2: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
    }),
  ),
};

InsightsPanel.defaultProps = {
  summary: null,
  events: [],
};
