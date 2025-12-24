import PropTypes from "prop-types";

const FALLBACK_GREENS = {
  Север: 0.45,
  Юг: 0.32,
  Восток: 0.27,
  Запад: 0.38,
};

export function PhaseVisuals({ summary, queueHistory }) {
  const greens = summary?.greens && Object.keys(summary.greens).length ? summary.greens : FALLBACK_GREENS;
  const greenValues = Object.values(greens);
  const avgGreen = greenValues.length ? greenValues.reduce((sum, value) => sum + value, 0) / greenValues.length : 0;

  const recentQueues = (queueHistory ?? []).slice(-18);
  const heatmapMax = recentQueues.reduce((max, entry) => {
    const queues = Object.values(entry.queues ?? {});
    return Math.max(max, queues.length ? Math.max(...queues) : 0);
  }, 1);
  const heatmapCells = recentQueues.map((entry, index) => {
    const peak = Math.max(0, ...Object.values(entry.queues ?? {}));
    return {
      id: `${entry.frame}-${index}`,
      intensity: heatmapMax ? peak / heatmapMax : 0,
      frame: entry.frame,
    };
  });

  return (
    <div className="card visuals-card">
      <div className="visual-section">
        <div className="donut" style={{ background: `conic-gradient(#22c55e ${avgGreen * 360}deg, #cbd5f5 0deg)` }}>
          <div className="donut-center">
            <span className="muted">средняя доля зелёного</span>
            <strong>{Math.round(avgGreen * 100)}%</strong>
          </div>
        </div>
        <ul className="legend">
          {Object.entries(greens).map(([approach, value]) => (
            <li key={approach}>
              <span>{approach}</span>
              <strong>{Math.round(value * 100)}%</strong>
            </li>
          ))}
        </ul>
      </div>
      <div className="visual-section">
        <div className="heatmap-header">
          <p className="eyebrow">Очереди, 18 последних кадров</p>
          <span className="muted">чем ярче, тем длиннее очередь</span>
        </div>
        <div className="heatmap-grid">
          {heatmapCells.length ? (
            heatmapCells.map((cell) => (
              <span
                key={cell.id}
                className="heatmap-cell"
                title={`Кадр ${cell.frame}`}
                style={{ opacity: Math.max(0.25, cell.intensity) }}
              />
            ))
          ) : (
            <div className="chart-placeholder">Нет данных очереди</div>
          )}
        </div>
      </div>
    </div>
  );
}

PhaseVisuals.propTypes = {
  summary: PropTypes.shape({
    greens: PropTypes.object,
  }),
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      queues: PropTypes.object,
    }),
  ),
};

PhaseVisuals.defaultProps = {
  summary: null,
  queueHistory: [],
};
