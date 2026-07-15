import type { PageGeo } from '../iiif/pages';

interface PageInfoPanelProps {
  page: PageGeo;
  /** Volume directory name, e.g. "brooklyn_ny_1906_vol_6". */
  volume: string;
  onClose: () => void;
}

/**
 * Stats and debugger links for the page selected on the volume map.
 *
 * The links use the debugger's `?files=` deep-link convention, so they open
 * the page's streets or georef view in this same app.
 */
export function PageInfoPanel(props: PageInfoPanelProps) {
  const { page, volume, onClose } = props;
  const base = `data/${volume}/${page.pageKey}`;
  return (
    <div className="page-info-panel">
      <div className="page-info-header">
        <strong>{page.pageKey}</strong>
        <button type="button" onClick={onClose} title="Deselect page">
          ×
        </button>
      </div>
      <dl>
        <dt>Scale</dt>
        <dd>{page.scalePixelsPerFoot.toFixed(2)} px/ft</dd>
        <dt>Rotation</dt>
        <dd>{page.rotationDegrees.toFixed(1)}°</dd>
        <dt>Size</dt>
        <dd>
          {page.width} × {page.height} px
        </dd>
      </dl>
      <div className="page-info-links">
        <a href={`?files=${base}.jpg,${base}.streets.json`}>streets view</a>
        <a href={`?files=${base}.jpg,${base}.georef.json`}>georef view</a>
      </div>
    </div>
  );
}
