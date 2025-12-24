import PropTypes from "prop-types";

export function UploadPanel({ fileName, status, error, onFileChange, onSubmit, downloadLinks }) {
  return (
    <div className="card upload-panel">
      <div>
        <p className="eyebrow">Видео</p>
        <h3>Загрузите ролик перекрёстка</h3>
        <p className="muted">
          Поддерживаются mp4 и mov, длительность до 2 минут. После обработки появится аннотированное видео и JSON с
          событиями.
        </p>
      </div>
      <label className="upload-input">
        <input type="file" accept="video/*" onChange={onFileChange} disabled={status === "processing"} />
        <span>{fileName ?? "Выберите файл"}</span>
      </label>
      <button className="primary-btn" disabled={!fileName || status === "processing"} onClick={onSubmit} type="button">
        {status === "processing" ? "Обрабатываем..." : "Запустить анализ"}
      </button>
      {error && <p className="error-text">{error}</p>}
      {downloadLinks.length > 0 && (
        <div className="download-block">
          {downloadLinks.map((link) => (
            <a key={link.label} href={link.href} target="_blank" rel="noreferrer">
              ⬇ {link.label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

UploadPanel.propTypes = {
  fileName: PropTypes.string,
  status: PropTypes.string.isRequired,
  error: PropTypes.string,
  onFileChange: PropTypes.func.isRequired,
  onSubmit: PropTypes.func.isRequired,
  downloadLinks: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      href: PropTypes.string.isRequired,
    }),
  ),
};

UploadPanel.defaultProps = {
  fileName: null,
  error: "",
  downloadLinks: [],
};
