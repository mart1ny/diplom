import PropTypes from "prop-types";

const BUCKET_SIZE = 45; // frames per bucket for histogram

const formatFrameWindow = (start, size) => {
  const end = start + size;
  return `${start}–${end}`;
};

const buildBuckets = (events) => {
  const buckets = new Map();
  (events ?? []).forEach((event) => {
    const frame = Number.isFinite(event?.frame) ? Number(event.frame) : 0;
    const bucketKey = Math.floor(frame / BUCKET_SIZE);
    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, {
        frameStart: bucketKey * BUCKET_SIZE,
        count: 0,
        maxRisk: 0,
      });
    }
    const bucket = buckets.get(bucketKey);
    bucket.count += 1;
    bucket.maxRisk = Math.max(bucket.maxRisk, Number(event?.risk_score ?? 0));
  });
  return Array.from(buckets.values()).sort((a, b) => a.frameStart - b.frameStart);
};

export function EventTimeline({ events, framesProcessed, planHistory }) {
  const buckets = buildBuckets(events);
  const maxCount = Math.max(1, ...buckets.map((bucket) => bucket.count));
  const totalEvents = events?.length ?? 0;
  const latestPlan = planHistory?.length ? planHistory[planHistory.length - 1] : null;

  return (
    <div className="card event-timeline">
      <div className="chart-header">
        <p className="eyebrow">Near-miss</p>
        <h3>Активность по кадрам</h3>
        <p className="muted">Гистограмма риска по окнам {BUCKET_SIZE} кадров.</p>
      </div>
      {buckets.length ? (
        <div className="timeline-bars">
          {buckets.map((bucket) => (
            <div
              key={bucket.frameStart}
              className="timeline-bar"
              title={`Кадры ${formatFrameWindow(bucket.frameStart, BUCKET_SIZE)} · событий ${bucket.count}`}
              style={{
                height: `${(bucket.count / maxCount) * 100}%`,
                background: `linear-gradient(180deg, rgba(248,113,113,${
                  bucket.maxRisk || 0.2
                }), rgba(59,130,246,0.8))`,
              }}
            >
              <span>{bucket.count}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="chart-placeholder">События появятся после анализа.</div>
      )}
      <div className="chart-inline-stats">
        <div>
          <span className="muted">Всего событий</span>
          <strong>{totalEvents}</strong>
        </div>
        <div>
          <span className="muted">Кадров обработано</span>
          <strong>{framesProcessed}</strong>
        </div>
        <div>
          <span className="muted">Последний цикл</span>
          <strong>{latestPlan?.plan?.cycle ? `${latestPlan.plan.cycle.toFixed(1)} c` : "—"}</strong>
        </div>
      </div>
    </div>
  );
}

EventTimeline.propTypes = {
  events: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      risk_score: PropTypes.number,
    }),
  ),
  framesProcessed: PropTypes.number,
  planHistory: PropTypes.arrayOf(
    PropTypes.shape({
      frame: PropTypes.number,
      plan: PropTypes.shape({
        cycle: PropTypes.number,
      }),
    }),
  ),
};

EventTimeline.defaultProps = {
  events: [],
  framesProcessed: 0,
  planHistory: [],
};
