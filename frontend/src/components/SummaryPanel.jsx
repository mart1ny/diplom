import PropTypes from "prop-types";

export function SummaryPanel({ summary, duration }) {
  if (!summary) {
    return (
      <div className="card">
        <p>Загрузите видео, чтобы увидеть метрики очередей и рисков.</p>
      </div>
    );
  }

  const { total_events, max_queue_by_approach = {}, latest_cycle, greens = {}, risk_peaks = {} } = summary;

  const dominantApproach =
    Object.entries(max_queue_by_approach).sort(([, a], [, b]) => b - a)[0]?.[0] ?? "—";

  return (
    <div className="card">
      <div className="summary-grid">
        <div className="summary-card">
          <h4>Near-miss события</h4>
          <strong>{total_events}</strong>
          <p className="chip">за весь ролик</p>
        </div>
        <div className="summary-card">
          <h4>Длина цикла</h4>
          <strong>{latest_cycle ? `${latest_cycle.toFixed(1)} c` : "—"}</strong>
          <p className="chip">последняя оптимизация</p>
        </div>
        <div className="summary-card">
          <h4>Лидер очереди</h4>
          <strong>{dominantApproach}</strong>
          <p className="chip">максимум {max_queue_by_approach[dominantApproach] ?? 0} авто</p>
        </div>
        <div className="summary-card">
          <h4>Обработано кадров</h4>
          <strong>{duration}</strong>
          <p className="chip">frame count</p>
        </div>
      </div>

      <div className="grid" style={{ marginTop: "1.5rem" }}>
        <div>
          <h4>Доли зелёного</h4>
          <div className="queue-bars">
            {Object.entries(greens).map(([approach, value]) => (
              <div className="queue-bar" key={approach} title={`${approach}: ${(value * 100).toFixed(1)}%`}>
                <span style={{ width: `${Math.min(100, value * 100)}%` }} />
              </div>
            ))}
          </div>
        </div>
        <div>
          <h4>Пики риска</h4>
          <div className="queue-bars">
            {Object.entries(risk_peaks).map(([approach, value]) => (
              <div className="queue-bar" key={approach} title={`${approach}: ${(value * 100).toFixed(1)}%`}>
                <span style={{ width: `${Math.min(100, value * 100)}%`, background: "#f97316" }} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

SummaryPanel.propTypes = {
  summary: PropTypes.shape({
    total_events: PropTypes.number,
    max_queue_by_approach: PropTypes.object,
    latest_cycle: PropTypes.number,
    greens: PropTypes.object,
    risk_peaks: PropTypes.object,
  }),
  duration: PropTypes.number,
};

SummaryPanel.defaultProps = {
  summary: null,
  duration: 0,
};
