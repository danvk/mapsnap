/**
 * What a town has: the years it was surveyed, and the volumes of one of them.
 *
 * The year list is the point of the panel. A Sanborn town is not one map but a
 * series of them, and which years exist -- and which of those the run has
 * actually placed -- is the first thing worth knowing about a place.
 */

import { isLoaded, type LoadedVolume, type MissingVolume } from './annotations';
import type { PageGeo } from '../iiif/pages';
import type { PageRef } from './AtlasMap';
import { volumesOfYear, yearsOf, type Place, type Volume } from './places';

interface VolumePanelProps {
  place: Place;
  /** Every catalogued volume of the town, or null while the state file loads. */
  volumes: Volume[] | null;
  year: number | null;
  onSelectYear: (year: number | null) => void;
  /** Load results for the selected year, in the order they were requested. */
  results: (LoadedVolume | MissingVolume)[];
  loading: boolean;
  selectedPage: PageRef | null;
  onClose: () => void;
}

/** The page a selection points at, when its volume finished loading. */
function selectedPageGeo(
  results: (LoadedVolume | MissingVolume)[],
  selected: PageRef | null,
): { page: PageGeo; volume: Volume } | null {
  if (!selected) return null;
  for (const result of results) {
    if (!isLoaded(result) || result.volume.item !== selected.item) continue;
    const page = result.pages.find((p) => p.itemIndex === selected.itemIndex);
    if (page) return { page, volume: result.volume };
  }
  return null;
}

/** "3 of 4 volumes placed" for the year on screen. */
function coverageLine(results: (LoadedVolume | MissingVolume)[]): string {
  const placed = results.filter(isLoaded).length;
  const sheets = results
    .filter(isLoaded)
    .reduce((sum, result) => sum + result.pages.length, 0);
  if (results.length === 0) return 'no volumes';
  return (
    `${placed} of ${results.length} volume${results.length === 1 ? '' : 's'} ` +
    `placed · ${sheets.toLocaleString()} sheets`
  );
}

export function VolumePanel(props: VolumePanelProps) {
  const {
    place,
    volumes,
    year,
    onSelectYear,
    results,
    loading,
    selectedPage,
    onClose,
  } = props;
  const selected = selectedPageGeo(results, selectedPage);
  const years = volumes ? yearsOf(volumes) : [];

  return (
    <div className="atlas-panel">
      <div className="atlas-panel-header">
        <div>
          <strong>{place.name}</strong>
          <div className="atlas-panel-sub">{place.state}</div>
        </div>
        <button type="button" onClick={onClose} title="Back to the map">
          ×
        </button>
      </div>

      <div className="atlas-panel-sub">
        {place.volumes} volumes · {place.sheets.toLocaleString()} sheets
        {place.firstYear !== null && (
          <>
            {' · '}
            {place.firstYear}–{place.lastYear}
          </>
        )}
      </div>

      {!volumes && <p className="atlas-note">Loading volumes…</p>}

      {volumes && (
        <>
          <div className="atlas-section-label">Years</div>
          <div className="atlas-years">
            {years.map((candidate) => {
              const ofYear = volumesOfYear(volumes, candidate);
              const mirrored = ofYear.filter((volume) => volume.state).length;
              return (
                <button
                  key={String(candidate)}
                  type="button"
                  className={
                    'atlas-year' +
                    (candidate === year ? ' is-current' : '') +
                    (mirrored === 0 ? ' is-empty' : '')
                  }
                  title={
                    mirrored === 0
                      ? `${ofYear.length} volume(s), none mirrored`
                      : `${mirrored} of ${ofYear.length} volume(s) mirrored`
                  }
                  onClick={() => onSelectYear(candidate)}
                >
                  {candidate ?? 'undated'}
                  {ofYear.length > 1 && (
                    <span className="atlas-year-count">{ofYear.length}</span>
                  )}
                </button>
              );
            })}
          </div>

          <div className="atlas-section-label">
            {year ?? 'Undated'}
            <span className="atlas-section-note">
              {loading ? 'loading…' : coverageLine(results)}
            </span>
          </div>
          <ul className="atlas-volumes">
            {results.map((result) => (
              <li key={result.volume.item}>
                <span className="atlas-volume-item">{result.volume.item}</span>
                <span
                  className={
                    'atlas-volume-status' + (isLoaded(result) ? '' : ' is-gap')
                  }
                >
                  {isLoaded(result)
                    ? `${result.pages.length} sheets`
                    : result.reason}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {selected && (
        <>
          <div className="atlas-section-label">Sheet</div>
          <dl className="atlas-page">
            <dt>Page</dt>
            <dd>{selected.page.stem}</dd>
            <dt>Volume</dt>
            <dd>{selected.volume.item}</dd>
            <dt>Scale</dt>
            <dd>{selected.page.scalePixelsPerFoot.toFixed(2)} px/ft</dd>
            <dt>Rotation</dt>
            <dd>{selected.page.rotationDegrees.toFixed(1)}°</dd>
            <dt>GCPs</dt>
            <dd>{selected.page.gcps.length}</dd>
          </dl>
          <a
            className="atlas-link"
            href={`https://www.loc.gov/item/${selected.volume.item}/`}
            target="_blank"
            rel="noreferrer"
          >
            View at the Library of Congress ↗
          </a>
        </>
      )}
    </div>
  );
}
