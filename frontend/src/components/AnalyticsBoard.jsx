import PropTypes from "prop-types";

const DEFAULT_APPROACHES = [
  { name: "Север", risk: 0.2, green: 0.45 },
  { name: "Юг", risk: 0.15, green: 0.38 },
  { name: "Запад", risk: 0.32, green: 0.27 },
];

const formatPercentage = (value) => `${(value * 100).toFixed(0)}%`;

export function AnalyticsBoard({ queueHistory, planHistory, summary }) {
  const queueTrend = (queueHistory ?? []).slice(-12);
  const queuePoints = queueTrend.map((entry) => ({
    frame: entry.frame,
    maxQueue: Math.max(0, ...Object.values(entry.queues ?? {})),
  }));
  const maxQueue = Math.max(1, ...queuePoints.map((point) => point.maxQueue));
  const sparkPath = queuePoints
    .map((point, index) => {
      const progress = queuePoints.length > 1 ? index / (queuePoints.length - 1) : 0;
      const x = progress * 100;
      const y = 60 - (point.maxQueue / maxQueue) * 50;
      return `${index === 0 ? "M" : "L"}${x},${Number.isFinite(y) ? y : 60}`;
    })
    .join(" ");

  const riskEntries = summary
    ? Object.keys({ ...(summary.risk_peaks ?? {}), ...(summary.greens ?? {}) }).map((approach) => ({
        name: approach,
        risk: summary.risk_peaks?.[approach] ?? 0,
        green: summary.greens?.[approach] ?? 0,
      }))
    : DEFAULT_APPROACHES;

  const latestPlans = (planHistory ?? []).slice(-6);

  return (
    <div className="grid analytics-grid">
      <div className="card chart-card">
        <div className="chart-header">
          <p className="eyebrow">Очереди</p>
          <h3>Динамика максимальных очередей</h3>
          <p className="muted">Обновляется по мере обработки кадров. Пик: {maxQueue} авто.</p>
        </div>
        {queuePoints.length > 1 ? (
          <svg className="sparkline" viewBox="0 0 100 60" preserveAspectRatio="none">
            <defs>
              <linearGradient id="queueGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#06b6d4" />
              </linearGradient>
            </defs>
            <path d={sparkPath} fill="none" stroke="url(#queueGradient)" strokeWidth="2" />
            {queuePoints.map((point, index) => {
              const progress = queuePoints.length > 1 ? index / (queuePoints.length - 1) : 0;
              const cx = progress * 100;
              const cy = 60 - (point.maxQueue / maxQueue) * 50;
              return <circle key={point.frame ?? index} cx={cx} cy={Number.isFinite(cy) ? cy : 60} r="1.5" fill="#6366f1" />;
            })}
          </svg>
        ) : (
          <div className="chart-placeholder">Ждём данные очередей...</div>
        )}
        <div className="chart-inline-stats">
          <div>
            <span className="muted">Последний кадр</span>
            <strong>{queuePoints.at(-1)?.frame ?? "—"}</strong>
          </div>
          <div>
            <span className="muted">Очередь</span>
            <strong>{queuePoints.at(-1)?.maxQueue ?? 0} авто</strong>
          </div>
        </div>
      </div>

      <div className="card chart-card">
        <div className="chart-header">
          <p className="eyebrow">Риски и зелёный</p>
          <h3>Профиль по подходам</h3>
          <p className="muted">Сравнение near-miss риска и доли зелёного.</p>
        </div>
        <div className="risk-rows">
          {riskEntries.map((entry) => (
            <div key={entry.name} className="risk-row">
              <div className="approach-name">{entry.name}</div>
              <div className="dual-bar">
                <span style={{ width: `${Math.min(entry.risk, 1) * 100}%` }} title={`Риск: ${formatPercentage(entry.risk)}`} />
                <span style={{ width: `${Math.min(entry.green, 1) * 100}%`, background: "#22c55e" }} title={`Зелёный: ${formatPercentage(entry.green)}`} />
              </div>
              <div className="risk-meta">
                <small>Риск {formatPercentage(entry.risk)}</small>
                <small>Зелёный {formatPercentage(entry.green)}</small>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card chart-card">
        <div className="chart-header">
          <p className="eyebrow">Фазы</p>
          <h3>Хронология оптимизаций</h3>
          <p className="muted">Последние расчёты циклов и их состояние.</p>
        </div>
        {latestPlans.length ? (
          <div className="timeline">
            {latestPlans.map((plan) => (
              <div key={plan.frame} className="timeline-item">
                <span className="dot" />
                <div>
                  <strong>Кадр {plan.frame}</strong>
                  <p>
                    Цикл {plan.plan?.cycle ? `${plan.plan.cycle.toFixed(1)} с` : "—"} · {plan.plan?.status ?? "без статуса"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="chart-placeholder">История планов появится после первого расчёта.</div>
        )}
      </div>
    </div>
  );
}

AnalyticsBoard.propTypes = {
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      queues: PropTypes.object,
    }),
  ),
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      plan: PropTypes.shape({ cycle: PropTypes.number, status: PropTypes.string }),
    }),
  ),
  summary: PropTypes.shape({
    greens: PropTypes.object,
    risk_peaks: PropTypes.object,
  }),
};

AnalyticsBoard.defaultProps = {
  queueHistory: [],
  planHistory: [],
  summary: null,
};
