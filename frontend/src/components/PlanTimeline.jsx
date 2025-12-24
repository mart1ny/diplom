import PropTypes from "prop-types";

export function PlanTimeline({ planHistory }) {
  const recent = (planHistory ?? []).slice(-6).reverse();

  return (
    <div className="card plan-timeline">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Планы фаз</p>
          <h3>История оптимизаций</h3>
        </div>
      </div>
      {recent.length ? (
        <ul>
          {recent.map((entry) => (
            <li key={entry.frame}>
              <div>
                <strong>Кадр {entry.frame}</strong>
                <p className="muted tiny">
                  Цикл: {entry.plan?.cycle ? `${entry.plan.cycle.toFixed(1)} c` : "—"} · Статус:{" "}
                  {entry.plan?.status ?? "—"}
                </p>
              </div>
              <div className="plan-badges">
                {entry.plan?.greens
                  ? Object.entries(entry.plan.greens)
                      .slice(0, 3)
                      .map(([approach, value]) => (
                        <span key={approach}>
                          {approach}: {(value * 100).toFixed(0)}%
                        </span>
                      ))
                  : null}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="placeholder">План появится после первой оптимизации.</div>
      )}
    </div>
  );
}

PlanTimeline.propTypes = {
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      plan: PropTypes.shape({
        cycle: PropTypes.number,
        status: PropTypes.string,
        greens: PropTypes.object,
      }),
    }),
  ),
};

PlanTimeline.defaultProps = {
  planHistory: [],
};
