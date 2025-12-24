import PropTypes from "prop-types";

const DEFAULT_POINT = { frame: 0, cycle: 0 };

export function CycleChart({ planHistory }) {
  const data = (planHistory ?? [])
    .map((entry) => ({
      frame: entry.frame,
      cycle: entry.plan?.cycle ?? 0,
    }))
    .filter((entry) => entry.cycle > 0);
  const points = data.length ? data.slice(-20) : [DEFAULT_POINT];
  const maxCycle = Math.max(1, ...points.map((point) => point.cycle));
  const minCycle = Math.min(...points.map((point) => point.cycle));
  const path = points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 100;
      const y = 100 - (point.cycle / maxCycle) * 90;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Цикл</p>
          <h3>Стабильность длины цикла</h3>
        </div>
      </div>
      {points.length > 1 ? (
        <svg className="line-chart" viewBox="0 0 100 100" preserveAspectRatio="none">
          <path d={path} />
          {points.map((point, index) => {
            const x = (index / Math.max(points.length - 1, 1)) * 100;
            const y = 100 - (point.cycle / maxCycle) * 90;
            return <circle key={point.frame ?? index} cx={x} cy={y} r={1.2} />;
          })}
        </svg>
      ) : (
        <div className="placeholder">Планов ещё не было.</div>
      )}
      <div className="chart-footer">
        <div>
          <p className="muted tiny">Мин</p>
          <strong>{minCycle.toFixed(1)}</strong>
        </div>
        <div>
          <p className="muted tiny">Макс</p>
          <strong>{maxCycle.toFixed(1)}</strong>
        </div>
      </div>
    </div>
  );
}

CycleChart.propTypes = {
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      plan: PropTypes.shape({
        cycle: PropTypes.number,
      }),
    }),
  ),
};

CycleChart.defaultProps = {
  planHistory: [],
};
