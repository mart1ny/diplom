import PropTypes from "prop-types";

const DIRECTIONS = [
  { key: "north", label: "Север" },
  { key: "east", label: "Восток" },
  { key: "south", label: "Юг" },
  { key: "west", label: "Запад" },
];

const STATUS_LABEL = {
  high: "Высокая нагрузка",
  medium: "Средняя нагрузка",
  low: "Свободно",
};

const classifyLoad = (current, peak) => {
  if (!peak) {
    return "low";
  }
  const ratio = current / peak;
  if (ratio >= 0.75) {
    return "high";
  }
  if (ratio >= 0.45) {
    return "medium";
  }
  return "low";
};

export function DirectionalLoad({ queueHistory }) {
  const history = queueHistory ?? [];
  const lastEntry = history.at(-1) ?? null;
  const prevEntry = history.length > 1 ? history.at(-2) : null;

  const cards = DIRECTIONS.map((direction) => {
    const series = history.map((entry) => Number(entry.queues?.[direction.key] ?? 0));
    const currentValue = Number(lastEntry?.queues?.[direction.key] ?? 0);
    const peakValue = series.length ? Math.max(...series) : currentValue;
    const avgValue = series.length ? series.reduce((sum, val) => sum + val, 0) / series.length : currentValue;
    const prevValue = Number(prevEntry?.queues?.[direction.key] ?? currentValue);
    const delta = currentValue - prevValue;
    return {
      key: direction.key,
      label: direction.label,
      currentValue,
      peakValue,
      avgValue,
      delta,
      loadClass: classifyLoad(currentValue, peakValue),
    };
  });

  const scaleMax = Math.max(1, ...cards.map((card) => Math.max(card.currentValue, card.peakValue)));
  const hasAnyData = history.length > 0;

  return (
    <div className="card directional-panel">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Подходы</p>
          <h3>Нагрузка по направлениям</h3>
        </div>
        <span className="badge muted tiny">
          обновляется по кадру: {history.at(-1)?.frame ?? "—"}
        </span>
      </div>
      {hasAnyData ? (
        <>
          <div className="direction-bars">
            {cards.map((card) => (
              <div className="direction-bar-column" key={card.key}>
                <div className="direction-bar-wrapper">
                  <div className="direction-bar-peak" style={{ height: `${(card.peakValue / scaleMax) * 100}%` }} />
                  <div
                    className={`direction-bar-fill ${card.loadClass}`}
                    style={{ height: `${(card.currentValue / scaleMax) * 100}%` }}
                    title={`Пик: ${card.peakValue.toFixed(1)}`}
                  />
                </div>
                <div className="direction-bar-meta">
                  <div className="direction-bar-value">{card.currentValue.toFixed(1)}</div>
                  <div className="direction-label">{card.label}</div>
                  <div className={`direction-delta ${card.delta >= 0 ? "trend-up" : "trend-down"}`}>
                    {card.delta >= 0 ? "+" : ""}
                    {card.delta.toFixed(1)} к предыдущему кадру
                  </div>
                  <span className={`direction-load ${card.loadClass}`}>{STATUS_LABEL[card.loadClass]}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="direction-meta">
            {cards.map((card) => (
              <div key={`${card.key}-meta`}>
                <p className="muted tiny">{card.label}: среднее за окно</p>
                <strong>{card.avgValue.toFixed(1)} авто</strong>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="placeholder">История очередей появится после обработки.</div>
      )}
    </div>
  );
}

DirectionalLoad.propTypes = {
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      queues: PropTypes.objectOf(PropTypes.number),
    }),
  ),
};

DirectionalLoad.defaultProps = {
  queueHistory: [],
};
