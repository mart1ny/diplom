import PropTypes from "prop-types";

const LEVELS = [
  { key: "high", label: "Высокий", color: "#dc2626" },
  { key: "medium", label: "Средний", color: "#d97706" },
  { key: "low", label: "Низкий", color: "#059669" },
];

export function RiskBreakdown({ events }) {
  const counts = LEVELS.map((level) => ({
    ...level,
    value: events.filter((event) => event.severity === level.key).length,
  }));
  const total = counts.reduce((sum, level) => sum + level.value, 0) || 1;

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Риск</p>
          <h3>Распределение событий</h3>
        </div>
      </div>
      <div className="risk-bars">
        {counts.map((level) => (
          <div key={level.key}>
            <div className="risk-bar">
              <span
                style={{
                  width: `${(level.value / total) * 100}%`,
                  background: level.color,
                }}
              />
            </div>
            <div className="risk-bar-meta">
              <strong>{level.label}</strong>
              <span>{level.value}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

RiskBreakdown.propTypes = {
  events: PropTypes.arrayOf(
    PropTypes.shape({
      severity: PropTypes.string,
    }),
  ),
};

RiskBreakdown.defaultProps = {
  events: [],
};
