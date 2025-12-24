import PropTypes from "prop-types";

const LEVEL_LABEL = {
  info: "INFO",
  warning: "ALERT",
  error: "ERROR",
  debug: "DEBUG",
};

const formatTimestamp = (timestamp) => {
  if (!timestamp) {
    return "—";
  }
  try {
    return new Date(timestamp * 1000).toLocaleTimeString("ru-RU", { hour12: false });
  } catch (err) {
    return "—";
  }
};

const formatDetails = (details) => {
  if (!details) {
    return null;
  }
  return Object.entries(details)
    .map(([key, value]) => {
      if (typeof value === "object") {
        try {
          const str = JSON.stringify(value);
          return `${key}: ${str.length > 80 ? `${str.slice(0, 77)}…` : str}`;
        } catch (err) {
          return `${key}: [object]`;
        }
      }
      return `${key}: ${value}`;
    })
    .join(" · ");
};

export function LogPanel({ logs }) {
  const latestLogs = (logs ?? []).slice(-12).reverse();

  return (
    <div className="card log-panel">
      <div className="panel-header">
        <p className="eyebrow">Логи</p>
        <h3>Ход обработки</h3>
        <p className="muted">Фактические события пайплайна за последние минуты.</p>
      </div>
      {latestLogs.length ? (
        <div className="log-list">
          {latestLogs.map((entry, index) => {
            const level = entry.level || "info";
            const levelLabel = LEVEL_LABEL[level] || level.toUpperCase();
            const detailsText = formatDetails(entry.details);
            return (
              <div key={`${entry.timestamp ?? index}-${index}`} className={`log-row log-${level}`}>
                <div>
                  <span className="log-level-label">{levelLabel}</span>
                  <p className="log-message">{entry.message}</p>
                  {detailsText && <small className="log-details">{detailsText}</small>}
                </div>
                <div className="log-meta">
                  {Number.isFinite(entry.frame) && <span>Кадр {entry.frame}</span>}
                  <span>{formatTimestamp(entry.timestamp)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="placeholder">Логи появятся после запуска пайплайна.</div>
      )}
    </div>
  );
}

LogPanel.propTypes = {
  logs: PropTypes.arrayOf(
    PropTypes.shape({
      level: PropTypes.string,
      message: PropTypes.string,
      timestamp: PropTypes.number,
      frame: PropTypes.number,
      details: PropTypes.object,
    }),
  ),
};

LogPanel.defaultProps = {
  logs: [],
};
