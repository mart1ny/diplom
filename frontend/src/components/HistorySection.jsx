import PropTypes from "prop-types";

const formatQueues = (queues = {}) =>
  Object.entries(queues)
    .map(([name, value]) => `${name}: ${value}`)
    .join(" · ");

export function HistorySection({ queueHistory, planHistory }) {
  if (!queueHistory?.length && !planHistory?.length) {
    return null;
  }

  const recentQueues = queueHistory.slice(-6).reverse();
  const recentPlans = planHistory.slice(-6).reverse();

  return (
    <div className="grid" style={{ marginTop: "1.5rem" }}>
      {recentQueues.length > 0 && (
        <div className="card">
          <h3>История очередей</h3>
          <table className="history-table">
            <thead>
              <tr>
                <th>Кадр</th>
                <th>Очереди</th>
              </tr>
            </thead>
            <tbody>
              {recentQueues.map((entry) => (
                <tr key={entry.frame}>
                  <td>{entry.frame}</td>
                  <td>{formatQueues(entry.queues)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {recentPlans.length > 0 && (
        <div className="card">
          <h3>Оптимизация фаз</h3>
          <table className="history-table">
            <thead>
              <tr>
                <th>Кадр</th>
                <th>Цикл</th>
                <th>Статус</th>
              </tr>
            </thead>
            <tbody>
              {recentPlans.map((entry) => (
                <tr key={entry.frame}>
                  <td>{entry.frame}</td>
                  <td>{entry.plan?.cycle ? entry.plan.cycle.toFixed(1) : "—"}</td>
                  <td>{entry.plan?.status ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

HistorySection.propTypes = {
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number.isRequired,
      queues: PropTypes.object,
    }),
  ),
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number.isRequired,
      plan: PropTypes.object,
    }),
  ),
};

HistorySection.defaultProps = {
  queueHistory: [],
  planHistory: [],
};
