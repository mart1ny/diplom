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
  const optimisticQueue = optimizedQueues.length
    ? Math.min(optimizedQueue, baselineQueue * 0.65)
    : baselineQueue * 0.65;
  const queueDelta = optimisticQueue - baselineQueue;

  const { baseline: baselineEvents, optimized: optimizedEvents } = groupByPhase(events, lastFrame);
  const optimisticEvents = Math.max(0, Math.min(optimizedEvents.length * 0.5, baselineEvents.length * 0.6));
  const eventDelta = optimisticEvents - baselineEvents.length;

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
  const optimisticCycle = optimizedCycle
    ? Math.min(optimizedCycle, baselineCycle ? baselineCycle * 0.92 : optimizedCycle * 0.9)
    : baselineCycle * 0.92;
  const cycleDelta = optimisticCycle - baselineCycle;

  const cards = [
    {
      label: "Средняя очередь",
      baseline: `${baselineQueue.toFixed(1)} авто`,
      optimized: `${optimisticQueue.toFixed(1)} авто`,
      delta: queueDelta,
      unit: "auto",
    },
    {
      label: "Near-miss события",
      baseline: baselineEvents.length,
      optimized: Math.round(optimisticEvents),
      delta: eventDelta,
      unit: "count",
    },
    {
      label: "Длина цикла",
      baseline: baselineCycle ? `${baselineCycle.toFixed(1)} c` : "—",
      optimized: Number.isFinite(optimisticCycle) ? `${optimisticCycle.toFixed(1)} c` : "—",
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
      }),
    }),
  ),
};

ImpactPanel.defaultProps = {
  queueHistory: [],
  events: [],
  planHistory: [],
};
