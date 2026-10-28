/**
 * The selected volume: its editions, the one on show, and its sheets.
 *
 * A Sanborn town is not one map but a series of them. A volume (Chicago's
 * vol. 5, or a small town's only one) was surveyed and republished every few
 * years, and its edition buttons are the point of the panel. Below them is
 * the selected edition: links to the item viewer and loc.gov, the picked
 * sheet and how confident the run is about it, and the images the run could
 * not place, as thumbnails. For a town with nothing placed, that last is all
 * there is.
 */

import {
  isLoaded,
  pageReport,
  type LoadedVolume,
  type MissingVolume,
} from './annotations';
import type { PageRef } from './AtlasMap';
import {
  groupLabel,
  yearSpan,
  type EditionOption,
  type VolumeGroup,
} from './footprints';
import {
  itemViewerUrl,
  sheetOfStem,
  unplacedImages,
  type ItemRecord,
} from './itemRecord';
import { locItemUrl, locSheetUrl, type Place, type Volume } from './places';
import { UnplacedImages } from './UnplacedImages';

interface VolumePanelProps {
  place: Place;
  /** The town's volumes, or null while its files load. */
  groups: VolumeGroup[] | null;
  /** The selected volume, or null when the town has none. */
  group: VolumeGroup | null;
  /** The selected volume's editions as buttons. */
  editions: EditionOption[];
  /** The selected edition. */
  edition: Volume | null;
  /** Its annotation, or why there is none; null while loading. */
  result: LoadedVolume | MissingVolume | null;
  /** Its item record; undefined while loading, null when there is none. */
  record: ItemRecord | null | undefined;
  /** The other volumes of the town covering the selected spot. */
  alsoHere: VolumeGroup[];
  /** `atSpot`: picked as another volume covering the selected spot. */
  onSelectGroup: (group: VolumeGroup, options: { atSpot: boolean }) => void;
  onSelectEdition: (item: string) => void;
  selectedPage: PageRef | null;
  onClose: () => void;
}

/**
 * "44/67 images placed from 42 sheets" for one edition.
 *
 * Three different numbers: a run cuts a sheet that holds several maps into
 * panels and georeferences each separately, so Mansfield 1921 is 42 physical
 * sheets, 67 images after the cut, and 44 of those images placed. Sheets come
 * from the catalog, the other two from the annotation's own report.
 */
function editionCoverage(result: LoadedVolume): string {
  const images =
    result.totalImages === null
      ? `${result.pages.length} images placed`
      : `${result.pages.length}/${result.totalImages} images placed`;
  return `${images} from ${result.volume.sheets.toLocaleString()} sheets`;
}

/** Why an edition draws nothing, in the panel's words. */
function missingText(result: MissingVolume): string {
  switch (result.reason) {
    case 'not digitized':
      return 'not digitized: on paper only at the Library of Congress';
    case 'nothing placed':
      return `none of its ${result.volume.sheets.toLocaleString()} sheets could be placed`;
    case 'unreadable':
      return "couldn't read its annotation";
  }
}

/**
 * An edition's date: the catalog's, when it agrees with the year.
 *
 * The year is the mirror's, which wins where the two disagree (see
 * build_places.py): sanborn01790_085 is catalogd 1906, and its sheets are
 * 1950 maps.
 */
export function editionDate(edition: Volume): string {
  if (edition.year === null) return edition.date || 'Undated';
  return edition.date.startsWith(String(edition.year))
    ? edition.date
    : String(edition.year);
}

/** An edition button's tooltip. */
function editionTitle(option: EditionOption): string {
  const date = editionDate(option.volume);
  switch (option.status) {
    case 'placed':
      return `${date}: ${option.volume.sheets} sheets`;
    case 'unplaced':
      return `${date}: digitized, but none of its sheets could be placed`;
    case 'not digitized':
      return `${date}: on paper only at the Library of Congress`;
  }
}

/** A tint for a P(good): green when the run is sure, amber, then red. */
function confidenceClass(confidence: number): string {
  if (confidence >= 0.8) return 'is-high';
  if (confidence >= 0.5) return 'is-medium';
  return 'is-low';
}

/** The page a selection points at, with the service it is drawn from (which names its LoC sheet). */
function selectedPageOf(
  result: LoadedVolume | MissingVolume | null,
  selected: PageRef | null,
) {
  if (!selected || !result || !isLoaded(result)) return null;
  if (result.volume.item !== selected.item) return null;
  const page = result.pages.find((p) => p.itemIndex === selected.itemIndex);
  if (!page) return null;
  return {
    page,
    volume: result.volume,
    report: pageReport(result.annotation, page.itemIndex),
    serviceUrl: result.annotation.items?.[page.itemIndex]?.target?.source?.id,
  };
}

export function VolumePanel(props: VolumePanelProps) {
  const {
    place,
    groups,
    group,
    editions,
    edition,
    result,
    record,
    alsoHere,
    onSelectGroup,
    onSelectEdition,
    selectedPage,
    onClose,
  } = props;
  const selected = selectedPageOf(result, selectedPage);
  const sheet =
    selected && record ? sheetOfStem(record, selected.page.stem) : null;
  const unplaced = record ? unplacedImages(record) : [];
  const groupCount = groups?.length ?? 0;
  const label = (g: VolumeGroup) => groupLabel(g, place.name, groupCount);

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
        {place.volumes} catalogd volumes · {place.sheets.toLocaleString()}{' '}
        sheets
        {place.firstYear !== null && (
          <>
            {' · '}
            {place.firstYear}–{place.lastYear}
          </>
        )}
      </div>

      {!groups && <p className="atlas-note">Loading…</p>}

      {groups && groupCount > 1 && (
        <details className="atlas-groups" open={groupCount <= 12}>
          <summary className="atlas-section-label">
            {groupCount} volumes of {place.name}
          </summary>
          <div className="atlas-chips">
            {groups.map((g) => (
              <button
                key={g.index}
                type="button"
                className={
                  'atlas-chip' + (g.index === group?.index ? ' is-current' : '')
                }
                title={`${label(g)}: ${g.editions.length} edition${g.editions.length === 1 ? '' : 's'}, ${yearSpan(g.editions)}`}
                onClick={() => onSelectGroup(g, { atSpot: false })}
              >
                {g.number ? `Vol. ${g.number}` : label(g)}
              </button>
            ))}
          </div>
        </details>
      )}

      {group && (
        <>
          <div className="atlas-volume-title">{label(group)}</div>
          {group.overview && (
            <p className="atlas-note">
              A small-scale atlas of the whole city. It covers every other
              volume, so clicking the map never selects it; it is reached only
              from this list.
            </p>
          )}
          <div className="atlas-section-label">
            {editions.length === 1 ? 'Edition' : `${editions.length} editions`}
          </div>
          <div className="atlas-years">
            {editions.map((option) => (
              <button
                key={option.volume.item}
                type="button"
                className={
                  'atlas-year' +
                  (option.volume.item === edition?.item ? ' is-current' : '') +
                  (option.status === 'not digitized' ? ' is-empty' : '') +
                  (option.status === 'unplaced' ? ' is-unplaced' : '')
                }
                title={editionTitle(option)}
                disabled={option.status === 'not digitized'}
                onClick={() => onSelectEdition(option.volume.item)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </>
      )}

      {alsoHere.length > 0 && (
        <>
          <div className="atlas-section-label">Also covering this spot</div>
          <div className="atlas-chips">
            {alsoHere.map((g) => (
              <button
                key={g.index}
                type="button"
                className="atlas-chip"
                onClick={() => onSelectGroup(g, { atSpot: true })}
                title={`${g.editions.length} edition${g.editions.length === 1 ? '' : 's'}`}
              >
                {label(g)}
                {g.number && (
                  <span className="atlas-dim"> {yearSpan(g.editions)}</span>
                )}
              </button>
            ))}
          </div>
        </>
      )}
      {edition && (
        <div className="atlas-edition">
          <div className="atlas-section-label">
            {editionDate(edition)}
            <span className="atlas-section-note">
              {!result
                ? 'loading…'
                : isLoaded(result)
                  ? editionCoverage(result)
                  : missingText(result)}
            </span>
          </div>
          <div className="atlas-links">
            {edition.state && (
              <a
                className="atlas-link"
                href={itemViewerUrl(edition.item)}
                target="_blank"
                rel="noreferrer"
                title="Every sheet of this edition, and what the run did with each"
              >
                Item viewer ↗
              </a>
            )}
            <a
              className="atlas-link"
              href={locItemUrl(edition.item)}
              target="_blank"
              rel="noreferrer"
              title="This edition at the Library of Congress"
            >
              loc.gov ↗
            </a>
            <code className="atlas-item-id">{edition.item}</code>
          </div>
        </div>
      )}

      {selected && (
        <>
          <div className="atlas-section-label">Sheet</div>
          <dl className="atlas-page">
            <dt>Page</dt>
            <dd>
              {selected.page.stem}
              {sheet && <> (sheet {sheet.seq})</>}
            </dd>
            {selected.report.confidence !== null && (
              <>
                <dt title="The confidence model's estimate that this placement is within 25 ft of the truth">
                  P(good)
                </dt>
                <dd>
                  <span
                    className={
                      'atlas-confidence ' +
                      confidenceClass(selected.report.confidence)
                    }
                  >
                    {Math.round(selected.report.confidence * 100)}%
                  </span>
                  {selected.report.provisional && (
                    <span
                      className="atlas-dim"
                      title="Panels of split sheets: the model ranks them less well than whole pages"
                    >
                      {' '}
                      provisional
                    </span>
                  )}
                </dd>
              </>
            )}
            {selected.report.streets !== null && (
              <>
                <dt>Streets</dt>
                <dd>
                  {selected.report.streets} matched
                  {selected.report.intersections !== null &&
                    `, ${selected.report.intersections} intersections`}
                </dd>
              </>
            )}
            <dt>GCPs</dt>
            <dd>{selected.page.gcps.length}</dd>
            <dt>Scale</dt>
            <dd>{selected.page.scalePixelsPerFoot.toFixed(2)} px/ft</dd>
            <dt>Rotation</dt>
            <dd>{selected.page.rotationDegrees.toFixed(1)}°</dd>
            {selected.report.run && (
              <>
                <dt>Run</dt>
                <dd>
                  {selected.report.run}
                  {selected.report.model &&
                    selected.report.model !== selected.report.run &&
                    ` (model ${selected.report.model})`}
                </dd>
              </>
            )}
          </dl>
          <div className="atlas-links">
            {edition && (
              <a
                className="atlas-link"
                href={itemViewerUrl(edition.item, sheet?.seq)}
                target="_blank"
                rel="noreferrer"
              >
                Item viewer ↗
              </a>
            )}
            <a
              className="atlas-link"
              href={locSheetUrl(selected.volume, selected.serviceUrl)}
              target="_blank"
              rel="noreferrer"
            >
              loc.gov ↗
            </a>
          </div>
        </>
      )}

      {edition && unplaced.length > 0 && (
        <>
          <div className="atlas-section-label">
            Not placed
            <span className="atlas-section-note">
              {unplaced.length} image{unplaced.length === 1 ? '' : 's'}
            </span>
          </div>
          <UnplacedImages item={edition.item} images={unplaced} />
        </>
      )}
    </div>
  );
}
