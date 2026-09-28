/**
 * The selected volume: what it is, and the same spot in other years.
 *
 * The year buttons are the point of the panel. A Sanborn town is not one map
 * but a series of them, and for the spot on screen the question is which
 * surveys cover it -- those volumes, not the town's whole year list, are what
 * the buttons offer (see yearOptions).
 */

import { isLoaded, type LoadedVolume, type MissingVolume } from './annotations';
import type { PageGeo } from '../iiif/pages';
import type { PageRef } from './AtlasMap';
import type { YearOption } from './footprints';
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
  townVolumes: Volume[] | null;
  /** The selected volume, or null when the town has none to select. */
  volume: Volume | null;
  /** The selected volume's annotation, or why there is none; null while loading. */
  result: LoadedVolume | MissingVolume | null;
  /** The selected spot's years (see yearOptions). */
  years: YearOption[];
  onSelectItem: (item: string) => void;
  selectedPage: PageRef | null;
  onClose: () => void;
}

/** The page a selection points at, with the service it is drawn from (which names its LoC sheet). */
function selectedPageGeo(
  result: LoadedVolume | MissingVolume | null,
  selected: PageRef | null,
): { page: PageGeo; volume: Volume; serviceUrl: string | undefined } | null {
  if (!selected || !result || !isLoaded(result)) return null;
  if (result.volume.item !== selected.item) return null;
  const page = result.pages.find((p) => p.itemIndex === selected.itemIndex);
  if (!page) return null;
  const serviceUrl =
    result.annotation.items?.[page.itemIndex]?.target?.source?.id;
  return { page, volume: result.volume, serviceUrl };
}

/**
 * "44/67 images placed from 42 sheets" for one volume.
 *
 * Three different numbers: a run cuts a sheet that holds several maps into
 * panels and georeferences each separately, so Mansfield 1921 is 42 physical
 * sheets, 67 images after the cut, and 44 of those images placed. Sheets come
 * from the catalogue, the other two from the annotation's own report.
 */
function volumeCoverage(result: LoadedVolume): string {
  const images =
    result.totalImages === null
      ? `${result.pages.length} images placed`
      : `${result.pages.length}/${result.totalImages} images placed`;
  return `${images} from ${result.volume.sheets.toLocaleString()} sheets`;
}

/** A year button's tooltip. */
function yearTitle(option: YearOption): string {
  switch (option.status) {
    case 'placed':
      return `${option.year ?? 'undated'}: a volume covering this spot`;
    case 'unplaced':
      return 'digitized, but none of its sheets could be placed, so where it covers is unknown';
    case 'not digitized':
      return `not digitized — ${option.count} volume${option.count === 1 ? '' : 's'} on paper only, covering parts of town not known here`;
  }
}

export function VolumePanel(props: VolumePanelProps) {
  const {
    place,
    townVolumes,
    volume,
    result,
    years,
    onSelectItem,
    selectedPage,
    onClose,
  } = props;
  const selected = selectedPageGeo(result, selectedPage);

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

      {volume && (
        <>
          <div className="atlas-section-label">
            {volume.date || volume.year || 'Undated'}
            <span className="atlas-section-note">
              {!result
                ? 'loading…'
                : isLoaded(result)
                  ? volumeCoverage(result)
                  : result.reason}
            </span>
          </div>
          <a
            className="atlas-volume-item"
            href={locItemUrl(volume.item)}
            target="_blank"
            rel="noreferrer"
            title="This volume at the Library of Congress"
          >
            {volume.item}
          </a>

          <div className="atlas-section-label">This spot in other years</div>
          <div className="atlas-years">
            {years.map((option) => (
              <button
                key={`${option.year}-${option.item}`}
                type="button"
                className={
                  'atlas-year' +
                  (option.item === volume.item ? ' is-current' : '') +
                  (option.status === 'not digitized' ? ' is-empty' : '') +
                  (option.status === 'unplaced' ? ' is-unplaced' : '')
                }
                title={yearTitle(option)}
                disabled={option.item === null}
                onClick={() => option.item && onSelectItem(option.item)}
              >
                {option.year ?? 'undated'}
              </button>
            ))}
          </div>
        </>
      )}

      {!volume && townVolumes && (
        <>
          <p className="atlas-note">
            None of this town&rsquo;s volumes has a placed sheet to show.
          </p>
          <div className="atlas-section-label">Years</div>
          <div className="atlas-years">
            {yearsOf(townVolumes).map((year) => {
              const ofYear = volumesOfYear(townVolumes, year);
              const digitized = ofYear.filter((v) => v.state).length;
              return (
                <button
                  key={String(year)}
                  type="button"
                  disabled
                  className={'atlas-year' + (digitized ? '' : ' is-empty')}
                  title={
                    digitized
                      ? `${digitized} of ${ofYear.length} digitized, none placed`
                      : `not digitized — ${ofYear.length} volume${ofYear.length === 1 ? '' : 's'} on paper only`
                  }
                >
                  {year ?? 'undated'}
                </button>
              );
            })}
          </div>
        </>
      )}

      {selected && (
        <>
          <div className="atlas-section-label">Sheet</div>
          <dl className="atlas-page">
            <dt>Page</dt>
            <dd>{selected.page.stem}</dd>
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
