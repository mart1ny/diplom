import PropTypes from "prop-types";

const severityLabel = {
  high: "Высокий",
  medium: "Средний",
  low: "Низкий",
};

export function EventList({ events }) {
  const sorted = [...(events ?? [])]
    .filter((event) => Number.isFinite(event?.risk_score))
    .sort((a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0))
    .slice(0, 8);

  return (
    <div className="card event-list">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Near-miss события</p>
          <h3>Последние рисковые сцены</h3>
        </div>
      </div>
      {sorted.length ? (
        <table>
          <thead>
            <tr>
              <th>Кадр</th>
              <th>Треки</th>
              <th>Риск</th>
              <th>Скорость сближения</th>
              <th>Статус</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((event) => (
              <tr key={`${event.frame}-${event.id1}-${event.id2}`}>
                <td>{event.frame}</td>
                <td>
                  {event.id1} vs {event.id2}
                </td>
                <td>{(event.risk_score * 100).toFixed(0)}%</td>
                <td>{event.closing_speed ? `${event.closing_speed.toFixed(1)} px/s` : "—"}</td>
                <td>
                  <span className={`severity ${event.severity || "low"}`}>
                    {severityLabel[event.severity] ?? severityLabel.low}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="placeholder">События появятся после запуска анализа.</div>
      )}
    </div>
  );
}

EventList.propTypes = {
  events: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      id1: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
      id2: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
      risk_score: PropTypes.number,
      closing_speed: PropTypes.number,
      severity: PropTypes.string,
    }),
  ),
};

EventList.defaultProps = {
  events: [],
};
