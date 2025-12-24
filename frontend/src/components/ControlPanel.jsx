import PropTypes from "prop-types";
import { INTERSECTIONS, SIGNALS } from "../constants/network";

export function ControlPanel({ selectedIntersection, selectedSignal, onIntersectionChange, onSignalChange }) {
  const activeIntersection = INTERSECTIONS.find((item) => item.value === selectedIntersection) ?? INTERSECTIONS[0];

  return (
    <div className="card control-panel">
      <div className="panel-header">
        <p className="eyebrow">Сценарий мониторинга</p>
        <h3>Выбор перекрёстка и светофора</h3>
        <p className="muted">Задайте контекст: данные обновятся после расчёта пайплайна.</p>
      </div>
      <div className="selector-grid">
        <div className="selector">
          <label htmlFor="intersection-select">Перекрёсток</label>
          <select
            id="intersection-select"
            value={selectedIntersection}
            onChange={(event) => onIntersectionChange(event.target.value)}
          >
            {INTERSECTIONS.map((intersection) => (
              <option key={intersection.value} value={intersection.value}>
                {intersection.label} · {intersection.meta}
              </option>
            ))}
          </select>
          <div className="selector-cta">
            <span>{activeIntersection.flows}</span>
            <span>{activeIntersection.saturation}</span>
          </div>
        </div>
        <div className="selector">
          <label htmlFor="signal-select">Светофор</label>
          <select id="signal-select" value={selectedSignal} onChange={(event) => onSignalChange(event.target.value)}>
            {SIGNALS.map((signal) => (
              <option key={signal.value} value={signal.value}>
                {signal.label}
              </option>
            ))}
          </select>
          <div className="selector-cta">
            <span>Городской канал данных</span>
            <span className="status-pill soft">API готов</span>
          </div>
          <p className="selector-note">После интеграции система подключится к узлу и подтянет статистику автоматически.</p>
        </div>
      </div>
      <div className="panel-badges">
        <div>
          <strong>Данные</strong>
          <p>машинное зрение + трекинг объектов</p>
        </div>
        <div>
          <strong>Фаза</strong>
          <p>адаптивный цикл рассчитывает оптимизацию</p>
        </div>
        <div>
          <strong>Безопасность</strong>
          <p>near-miss события подсвечиваются в диаграммах</p>
        </div>
      </div>
    </div>
  );
}

ControlPanel.propTypes = {
  selectedIntersection: PropTypes.string.isRequired,
  selectedSignal: PropTypes.string.isRequired,
  onIntersectionChange: PropTypes.func.isRequired,
  onSignalChange: PropTypes.func.isRequired,
};
