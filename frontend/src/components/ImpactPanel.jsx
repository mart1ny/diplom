import PropTypes from "prop-types";

const computeAverageQueue = (entries) => {
  if (!entries.length) {
    return 0;
  }
  const sums = entries.map((entry) => {
    const values = Object.values(entry.queues ?? {});
    if (!values.length) {
      return 0;
    }
    return values.reduce((sum, val) => sum + Number(val || 0), 0) / values.length;
  });
  return sums.reduce((sum, val) => sum + val, 0) / sums.length;
};

const groupByPhase = (items, totalFrames) => {
  if (!totalFrames) {
    return { baseline: items, optimized: [] };
  }
  const midpoint = totalFrames / 2;
  const baseline = items.filter((item) => (item.frame ?? 0) <= midpoint);
  const optimized = items.filter((item) => (item.frame ?? 0) > midpoint);
  if (!baseline.length && optimized.length) {
    return { baseline: optimized.slice(0, 1), optimized };
  }
  if (!optimized.length && baseline.length) {
    return { baseline, optimized: baseline.slice(-1) };
  }
  return { baseline, optimized };
};

export function ImpactPanel({ queueHistory, events, planHistory }) {
  const lastFrame = queueHistory.at(-1)?.frame ?? planHistory.at(-1)?.frame ?? 0;
  const { baseline: baselineQueues, optimized: optimizedQueues } = groupByPhase(queueHistory, lastFrame);
  const baselineQueue = computeAverageQueue(baselineQueues);
  const optimizedQueue = computeAverageQueue(optimizedQueues);
  const queueDelta = optimizedQueue - baselineQueue;

  const { baseline: baselineEvents, optimized: optimizedEvents } = groupByPhase(events, lastFrame);
  const eventDelta = optimizedEvents.length - baselineEvents.length;

  const { baseline: baselinePlans, optimized: optimizedPlans } = groupByPhase(planHistory, lastFrame);
  const avgCycle = (list) => {
    const values = list.map((entry) => entry.plan?.cycle).filter((value) => Number.isFinite(value));
    if (!values.length) {
      return 0;
    }
    return values.reduce((sum, val) => sum + Number(val), 0) / values.length;
  };
  const baselineCycle = avgCycle(baselinePlans);
  const optimizedCycle = avgCycle(optimizedPlans);
  const cycleDelta = optimizedCycle - baselineCycle;
  const latestPlan = planHistory.at(-1)?.plan ?? null;
  const latestDurations = latestPlan?.durations
    ? Object.entries(latestPlan.durations).sort(([, a], [, b]) => b - a)
    : [];

  const cards = [
    {
      label: "Средняя очередь",
      baseline: `${baselineQueue.toFixed(1)} авто`,
      optimized: `${optimizedQueue.toFixed(1)} авто`,
      delta: queueDelta,
      unit: "auto",
    },
    {
      label: "Near-miss события",
      baseline: baselineEvents.length,
      optimized: optimizedEvents.length,
      delta: eventDelta,
      unit: "count",
    },
    {
      label: "Длина цикла",
      baseline: baselineCycle ? `${baselineCycle.toFixed(1)} c` : "—",
      optimized: Number.isFinite(optimizedCycle) && optimizedCycle > 0 ? `${optimizedCycle.toFixed(1)} c` : "—",
      delta: cycleDelta,
      unit: "sec",
    },
  ];

  const formatDelta = (value, unit) => {
    if (!Number.isFinite(value) || Math.abs(value) < 1e-2) {
      return "без изменений";
    }
    const formatted =
      unit === "sec"
        ? `${value > 0 ? "+" : ""}${value.toFixed(1)} c`
        : `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
    return value < 0 ? `${formatted} (меньше)` : `${formatted} (больше)`;
  };

  return (
    <div className="card impact-panel">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Влияние оптимизации</p>
          <h3>До vs после</h3>
        </div>
      </div>
      <div className="impact-grid">
        {cards.map((card) => (
          <div key={card.label} className="impact-item">
            <p className="muted tiny">{card.label}</p>
            <div className="impact-values">
              <div>
                <span>Было</span>
                <strong>{card.baseline}</strong>
              </div>
              <div>
                <span>После модели</span>
                <strong>{card.optimized}</strong>
              </div>
            </div>
            <p className="delta">{formatDelta(card.delta, card.unit)}</p>
          </div>
        ))}
      </div>
      <div className="chart-footer">
        <div>
          <p className="muted tiny">Optimizer</p>
          <strong>{latestPlan?.optimizer ?? "—"}</strong>
        </div>
        <div>
          <p className="muted tiny">Solver</p>
          <strong>{latestPlan?.solver_status ?? "—"}</strong>
        </div>
        <div>
          <p className="muted tiny">Objective</p>
          <strong>{Number.isFinite(latestPlan?.objective_value) ? Number(latestPlan.objective_value).toFixed(2) : "—"}</strong>
        </div>
        <div>
          <p className="muted tiny">Фазы</p>
          <strong>
            {latestDurations.length
              ? latestDurations.map(([approach, value]) => `${approach} ${Number(value).toFixed(1)}c`).join(" · ")
              : "—"}
          </strong>
        </div>
      </div>
    </div>
  );
}

ImpactPanel.propTypes = {
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      queues: PropTypes.object,
    }),
  ),
  events: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
    }),
  ),
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      plan: PropTypes.shape({
        cycle: PropTypes.number,
        optimizer: PropTypes.string,
        solver_status: PropTypes.string,
        objective_value: PropTypes.number,
        durations: PropTypes.object,
      }),
    }),
  ),
};

ImpactPanel.defaultProps = {
  queueHistory: [],
  events: [],
  planHistory: [],
};
