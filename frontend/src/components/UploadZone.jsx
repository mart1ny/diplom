import PropTypes from "prop-types";

export function UploadZone({ onFileChange, onSubmit, disabled, fileName, status }) {
  return (
    <div className="card">
      <div className="upload-zone">
        <input type="file" id="video-input" accept="video/*" onChange={onFileChange} disabled={disabled} />
        <label htmlFor="video-input">
          <strong>{fileName ?? "Выберите файл с перекрёстка"}</strong>
          <small>Поддерживаются mp4, mov и другие форматы видео</small>
        </label>
      </div>
      <button className="primary-btn" disabled={disabled || !fileName} onClick={onSubmit}>
        {status === "processing" ? "Обработка..." : "Загрузить и обработать"}
      </button>
      <div className="status-pill">Статус: <strong>{status === "idle" ? "Ожидание" : status === "processing" ? "Обработка" : "Готово"}</strong></div>
    </div>
  );
}

UploadZone.propTypes = {
  onFileChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  fileName: PropTypes.string,
  status: PropTypes.oneOf(["idle", "processing", "done"]),
};

UploadZone.defaultProps = {
  disabled: false,
  fileName: null,
  status: "idle",
};
