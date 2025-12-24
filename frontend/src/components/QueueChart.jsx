import PropTypes from "prop-types";

const defaultPoint = { frame: 0, value: 0 };

export function QueueChart({ queueHistory, events }) {
  const trend = (queueHistory ?? []).map((entry) => {
    const buckets = Object.values(entry.queues ?? {});
    const avg = buckets.length ? buckets.reduce((sum, val) => sum + Number(val || 0), 0) / buckets.length : 0;
    return { frame: entry.frame, value: avg };
  });
  const points = trend.length > 1 ? trend.slice(-60) : [defaultPoint];
  const maxValue = Math.max(1, ...points.map((p) => p.value));
  const nearMissCount = events?.length ?? 0;
  const chartWidth = 640;
  const chartHeight = 260;
  const margin = { top: 24, right: 24, bottom: 36, left: 56 };
  const innerWidth = chartWidth - margin.left - margin.right;
  const innerHeight = chartHeight - margin.top - margin.bottom;

  const getX = (index) => margin.left + (index / Math.max(points.length - 1, 1)) * innerWidth;
  const getY = (value) => margin.top + innerHeight - (value / maxValue) * innerHeight;

  const normalizedPoints =
    points.length > 1
      ? points.map((point, index) => ({
          x: getX(index),
          y: getY(point.value),
          frame: point.frame ?? index,
          value: point.value,
        }))
      : [];

  const linePath = normalizedPoints
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
    .join(" ");
  const areaPath = normalizedPoints.length
    ? `${linePath} L${normalizedPoints.at(-1).x.toFixed(2)},${(margin.top + innerHeight).toFixed(
        2,
      )} L${normalizedPoints[0].x.toFixed(2)},${(margin.top + innerHeight).toFixed(2)} Z`
    : "";

  const yTickCount = 5;
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, idx) => {
    const value = (maxValue / yTickCount) * idx;
    return {
      value,
      y: getY(value),
    };
  });

  const xTickCount = Math.min(6, points.length);
  const xTicks =
    points.length > 1
      ? Array.from({ length: xTickCount }, (_, idx) => {
          const targetIndex = Math.round((idx / Math.max(xTickCount - 1, 1)) * (points.length - 1));
          return {
            x: getX(targetIndex),
            frame: points[targetIndex]?.frame ?? targetIndex,
          };
        }).filter(
          (tick, index, arr) => index === arr.findIndex((item) => item.frame === tick.frame && item.x === tick.x),
        )
      : [];
  const lastPoint = normalizedPoints.at(-1) ?? null;

  return (
    <div className="card chart-card">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Очереди</p>
          <h3>Средний размер очереди</h3>
        </div>
        <span className="badge">Near-miss: {nearMissCount}</span>
      </div>
      {points.length > 1 ? (
        <svg className="queue-chart axis-chart" viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img">
          <desc>Средний размер очереди по кадрам</desc>
          <defs>
            <linearGradient id="queueGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.7" />
              <stop offset="100%" stopColor="#60a5fa" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect
            x={margin.left}
            y={margin.top}
            width={innerWidth}
            height={innerHeight}
            className="chart-background"
            rx="8"
            ry="8"
          />
          <g className="axis-grid">
            {yTicks.map((tick) => (
              <g key={`y-${tick.value}`}>
                <line x1={margin.left} x2={chartWidth - margin.right} y1={tick.y} y2={tick.y} className="grid-line" />
                <text x={margin.left - 10} y={tick.y + 4} className="axis-tick">
                  {tick.value.toFixed(0)}
                </text>
              </g>
            ))}
            {xTicks.map((tick) => (
              <line
                key={`x-grid-${tick.frame}`}
                x1={tick.x}
                x2={tick.x}
                y1={margin.top}
                y2={margin.top + innerHeight}
                className="grid-line vertical"
              />
            ))}
          </g>
          <g className="axis-baseline">
            <line
              x1={margin.left}
              x2={chartWidth - margin.right}
              y1={margin.top + innerHeight}
              y2={margin.top + innerHeight}
            />
            <line x1={margin.left} x2={margin.left} y1={margin.top} y2={margin.top + innerHeight} />
          </g>
          <g className="axis-x">
            {xTicks.map((tick) => (
              <g key={`x-${tick.frame}`}>
                <line
                  x1={tick.x}
                  x2={tick.x}
                  y1={margin.top + innerHeight}
                  y2={margin.top + innerHeight + 6}
                />
                <text x={tick.x} y={margin.top + innerHeight + 18} className="axis-tick x">
                  {tick.frame}
                </text>
              </g>
            ))}
          </g>
          <text className="axis-label axis-label-y" x={margin.left - 40} y={margin.top - 8}>
            авто
          </text>
          <text className="axis-label axis-label-x" x={chartWidth - margin.right - 32} y={chartHeight - 6}>
            кадр
          </text>
          {areaPath && <path className="area-path" d={areaPath} />}
          <path className="line-path" d={linePath} />
          {normalizedPoints.map((point) => (
            <circle key={point.frame} cx={point.x} cy={point.y} r={2} />
          ))}
          {lastPoint && (
            <g className="last-point">
              <circle cx={lastPoint.x} cy={lastPoint.y} r={4.5} />
              <text x={lastPoint.x + 8} y={lastPoint.y - 8} className="last-point-label">
                {lastPoint.value.toFixed(1)} авто
              </text>
            </g>
          )}
        </svg>
      ) : (
        <div className="placeholder">Данные появятся после анализа.</div>
      )}
      <div className="chart-footer">
        <div>
          <p className="muted tiny">Пик</p>
          <strong>{Math.round(maxValue)}</strong>
        </div>
        <div>
          <p className="muted tiny">Последний кадр</p>
          <strong>{points.at(-1)?.frame ?? "—"}</strong>
        </div>
      </div>
    </div>
  );
}

QueueChart.propTypes = {
  queueHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      queues: PropTypes.object,
    }),
  ),
  events: PropTypes.arrayOf(
    PropTypes.shape({
      severity: PropTypes.string,
    }),
  ),
};

QueueChart.defaultProps = {
  queueHistory: [],
  events: [],
};
