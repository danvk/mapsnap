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
import {
  locItemUrl,
  locSheetUrl,
  volumesOfYear,
  yearsOf,
  type Place,
  type Volume,
} from './places';

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

/**
 * The page a selection points at, when its volume finished loading, with the
 * image service it is drawn from (which names its LoC sheet).
 */
function selectedPageGeo(
  results: (LoadedVolume | MissingVolume)[],
  selected: PageRef | null,
): { page: PageGeo; volume: Volume; serviceUrl: string | undefined } | null {
  if (!selected) return null;
  for (const result of results) {
    if (!isLoaded(result) || result.volume.item !== selected.item) continue;
    const page = result.pages.find((p) => p.itemIndex === selected.itemIndex);
    if (page) {
      const serviceUrl =
        result.annotation.items?.[page.itemIndex]?.target?.source?.id;
      return { page, volume: result.volume, serviceUrl };
    }
  }
  return null;
}

/**
 * "44/67 images placed from 42 sheets" for the year on screen.
 *
 * Three different numbers, and the panel used to call two of them "sheets".
 * A run cuts a sheet that holds several maps into panels and georeferences
 * each separately, so Mansfield 1921 is 42 physical sheets, 67 images after
 * the cut, and 44 of those images placed. Sheets come from the catalogue,
 * the other two from the annotation's own report.
 */
function coverageLine(results: (LoadedVolume | MissingVolume)[]): string {
  if (results.length === 0) return 'no volumes';
  const loaded = results.filter(isLoaded);
  const placed = loaded.reduce((sum, result) => sum + result.pages.length, 0);
  const sheets = results.reduce((sum, result) => sum + result.volume.sheets, 0);
  // An annotation with no report card cannot say how many images it declined
  // to place, and inventing a denominator would overstate the coverage.
  const reported = loaded.every((result) => result.totalImages !== null)
    ? loaded.reduce((sum, result) => sum + (result.totalImages ?? 0), 0)
    : null;
  const images =
    reported === null
      ? `${placed.toLocaleString()} images placed`
      : `${placed.toLocaleString()}/${reported.toLocaleString()} images placed`;
  const missing = results.length - loaded.length;
  const gap =
    missing > 0
      ? ` · ${missing} volume${missing === 1 ? '' : 's'} missing`
      : '';
  return `${images} from ${sheets.toLocaleString()} sheets${gap}`;
}

/** "44/67 images" for one volume, or just the placed count without a report. */
function volumeCoverage(result: LoadedVolume): string {
  return result.totalImages === null
    ? `${result.pages.length} images`
    : `${result.pages.length}/${result.totalImages} images`;
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
                      ? // No volume of this year is in the mirror, which for
                        // all but 0.1% of the catalogue means the Library
                        // never scanned it: there is no image to place.
                        `not digitized — ${ofYear.length} volume${ofYear.length === 1 ? '' : 's'} on paper only`
                      : `${mirrored} of ${ofYear.length} volume${ofYear.length === 1 ? '' : 's'} digitized`
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
                <a
                  className="atlas-volume-item"
                  href={locItemUrl(result.volume.item)}
                  target="_blank"
                  rel="noreferrer"
                  title="This volume at the Library of Congress"
                >
                  {result.volume.item}
                </a>
                <span
                  className={
                    'atlas-volume-status' + (isLoaded(result) ? '' : ' is-gap')
                  }
                >
                  {isLoaded(result) ? volumeCoverage(result) : result.reason}
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
            href={locSheetUrl(selected.volume, selected.serviceUrl)}
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
