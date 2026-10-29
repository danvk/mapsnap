/**
 * The atlas: the Sanborn collection as one map of the country.
 *
 * Zoomed out, every town the collection covers is a dot, sized by how much of
 * it there is. Zoomed in, the dots become volumes: one shape per volume of a
 * town (Chicago's vol. 5, or all of a small town), whatever editions it has.
 * A click selects the volume there, opens it on its newest edition covering
 * the spot, and lists its other editions. A town the run placed nothing of
 * keeps its dot, and a click on it lists its editions too, with their sheets
 * as thumbnails. One edition is drawn at a time, which keeps what is fetched
 * and warped to one volume's sheets and nudges viewing toward the zoom levels
 * where the scans look good.
 *
 * Fetches, in widening order of cost: the place index and the footprint index
 * once at startup; a town's footprint file when its bounds come into view; a
 * state's volume list when one of its towns is selected; and the selected
 * edition's annotation (from the published run, see annotations.ts) and item
 * record (see itemRecord.ts). Sheet tiles are pulled by Allmaps as it draws,
 * and only for what is on screen.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  inParallel,
  isLoaded,
  loadVolume,
  type ImageSource,
  type LoadedVolume,
  type MissingVolume,
} from './annotations';
import { isTypingTarget } from '../keyboard';
import {
  AtlasMap,
  FOOTPRINT_ZOOM,
  type MapTarget,
  type PageRef,
} from './AtlasMap';
import {
  defaultEdition,
  editionOptions,
  footprintFeatures,
  groupOfItem,
  multiPolygonBounds,
  seriesAt,
  townsInView,
  volumeGroups,
  type Bounds,
  type FootprintIndex,
  type TownFootprints,
  type VolumeGroup,
} from './footprints';
import { DATA_URL, fetchItemRecord, type ItemRecord } from './itemRecord';
import { nextOpacity } from './opacity';
import { PlaceSearch } from './PlaceSearch';
import { VolumePanel } from './VolumePanel';
import {
  placeOfVolume,
  stateSlug,
  statesByVolumeKey,
  volumeKey,
  type Place,
  type PlaceIndex,
  type Volume,
} from './places';

/** How many town footprint files to fetch at once. */
const FETCH_WIDTH = 6;
/** At most this many towns' footprints are fetched for one view. */
const MAX_TOWNS_PER_VIEW = 80;
/** How close to fly to a town with no shape to frame. */
const TOWN_ZOOM = 14;

/**
 * Where the atlas's own files are: the place index, the footprints and the
 * per-state volume lists. In development they are app/public/atlas; a
 * deployed build reads them from data.mapsnap.org (VITE_ATLAS_DATA_URL, see
 * vite.atlas.config.ts).
 */
const ATLAS_DATA = (
  (import.meta.env.VITE_ATLAS_DATA_URL as string | undefined) ??
  `${import.meta.env.BASE_URL}atlas`
).replace(/\/$/, '');

/** The selected town, volume and edition, and the spot they were picked at. */
interface Selection {
  place: string;
  /** Index into the town's volumeGroups. */
  group: number;
  item: string | null;
  /** The clicked spot, for the other volumes covering it; null when not picked on the map. */
  point: [number, number] | null;
}

/** A town's files: its catalog entries, and its footprints when it has any. */
interface TownData {
  volumes: Volume[];
  footprints: TownFootprints | null;
}

// The edition a page URL names, if any: ?volume=<item>.
function itemFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get('volume');
}

// ?data= points the item records at another copy of data.mapsnap.org, for
// testing a build before it is uploaded, as on mapsnap.org/item.
const DATA = (
  new URLSearchParams(window.location.search).get('data') ?? DATA_URL
).replace(/\/$/, '');

/** Record the selected edition in the page URL, keeping ?data=. */
function setUrlItem(item: string | null): void {
  const query = new URLSearchParams(window.location.search);
  query.delete('volume');
  if (item) query.set('volume', item);
  const search = query.toString();
  window.history.replaceState(
    null,
    '',
    search ? `?${search}` : window.location.pathname,
  );
}

export function AtlasApp() {
  const [index, setIndex] = useState<PlaceIndex | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [footprintIndex, setFootprintIndex] = useState<FootprintIndex | null>(
    null,
  );
  const [sizeBy, setSizeBy] = useState<'sheets' | 'volumes'>('sheets');
  // Chronoscope by default: loc.gov rate-limits long before a town-year's
  // worth of tiles is drawn (see annotations.ts).
  const [imageSource, setImageSource] = useState<ImageSource>('chronoscope');
  // Sheet opacity in percent, as in the volume viewer: a slider, and `p` to
  // step through 100/50/0 so the map underneath can be checked at a keypress.
  const [opacity, setOpacity] = useState(100);
  // Holding the space bar hides the sheets, as in Allmaps, to show the map
  // underneath; letting go, or leaving the window, brings them back.
  const [peeking, setPeeking] = useState(false);
  useEffect(() => {
    function onKeydown(event: KeyboardEvent): void {
      if (isTypingTarget(event.target)) return;
      if (event.key === 'p') setOpacity(nextOpacity);
      if (event.key === ' ') {
        // Not a page scroll, nor a click on the focused button.
        event.preventDefault();
        setPeeking(true);
      }
    }
    function onKeyup(event: KeyboardEvent): void {
      if (event.key !== ' ' || isTypingTarget(event.target)) return;
      event.preventDefault();
      setPeeking(false);
    }
    const stopPeeking = () => setPeeking(false);
    window.addEventListener('keydown', onKeydown);
    window.addEventListener('keyup', onKeyup);
    window.addEventListener('blur', stopPeeking);
    return () => {
      window.removeEventListener('keydown', onKeydown);
      window.removeEventListener('keyup', onKeyup);
      window.removeEventListener('blur', stopPeeking);
    };
  }, []);

  const [towns, setTowns] = useState<ReadonlyMap<string, TownFootprints>>(
    () => new Map(),
  );
  const [selection, setSelection] = useState<Selection | null>(null);
  const [townVolumes, setTownVolumes] = useState<Volume[] | null>(null);
  const [result, setResult] = useState<LoadedVolume | MissingVolume | null>(
    null,
  );
  const [record, setRecord] = useState<ItemRecord | null | undefined>(
    undefined,
  );
  const [selectedPage, setSelectedPage] = useState<PageRef | null>(null);
  const [target, setTarget] = useState<MapTarget | null>(null);
  const [zoom, setZoom] = useState(0);

  // Fetched files are kept for the session: towns get panned back to, and a
  // state's volume list is good for every town in it.
  const townCache = useRef(new Map<string, Promise<TownFootprints | null>>());
  const stateCache = useRef(
    new Map<string, Promise<Record<string, Volume[]> | null>>(),
  );
  // Volume key -> the states whose volumes files hold it, for a link that
  // names only a volume. Fetched once, and only when such a link is opened.
  const volumeStates = useRef<Promise<Map<string, string[]> | null> | null>(
    null,
  );
  const placeById = useMemo(
    () => new Map((index?.places ?? []).map((place) => [place.id, place])),
    [index],
  );
  const townsWithFootprints = useMemo(
    () => new Set(Object.keys(footprintIndex?.towns ?? {})),
    [footprintIndex],
  );
  // The towns with a shape on the map; the rest keep a dot when zoomed in.
  const townsWithShapes = useMemo(() => {
    const shapeless = new Set(footprintIndex?.shapeless ?? []);
    return new Set([...townsWithFootprints].filter((t) => !shapeless.has(t)));
  }, [footprintIndex, townsWithFootprints]);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch(`${ATLAS_DATA}/places.json`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setIndex((await response.json()) as PlaceIndex);
      } catch (error) {
        setIndexError(
          `${String(error)} — run scripts/atlas/build_places.py to generate it`,
        );
      }
    })();
    void (async () => {
      const response = await fetch(`${ATLAS_DATA}/footprints/index.json`);
      const index = response.ok
        ? ((await response.json()) as Partial<FootprintIndex>)
        : null;
      if (index?.run && index.towns) {
        setFootprintIndex(index as FootprintIndex);
      } else {
        setIndexError(
          'no volume footprints — run scripts/atlas/build_footprints.py to generate them',
        );
      }
    })();
  }, []);

  const loadTown = useCallback((place: string) => {
    let pending = townCache.current.get(place);
    if (!pending) {
      pending = (async () => {
        const response = await fetch(`${ATLAS_DATA}/footprints/${place}.json`);
        if (!response.ok) return null;
        const town = (await response.json()) as TownFootprints;
        setTowns((previous) => new Map(previous).set(place, town));
        return town;
      })();
      townCache.current.set(place, pending);
    }
    return pending;
  }, []);

  const loadStateVolumes = useCallback((slug: string) => {
    let pending = stateCache.current.get(slug);
    if (!pending) {
      pending = (async () => {
        const response = await fetch(`${ATLAS_DATA}/volumes/${slug}.json`);
        return response.ok
          ? ((await response.json()) as Record<string, Volume[]>)
          : null;
      })();
      stateCache.current.set(slug, pending);
    }
    return pending;
  }, []);

  /** Both of a town's files; a town with no footprints has no file to fetch. */
  const loadTownData = useCallback(
    async (place: string): Promise<TownData> => {
      const [byPlace, footprints] = await Promise.all([
        loadStateVolumes(stateSlug(place)),
        townsWithFootprints.has(place) ? loadTown(place) : null,
      ]);
      return { volumes: byPlace?.[place] ?? [], footprints };
    },
    [loadStateVolumes, loadTown, townsWithFootprints],
  );

  // The town a volume is in, from its state's volumes file.
  const resolveVolumePlace = useCallback(
    async (item: string): Promise<string | null> => {
      volumeStates.current ??= (async () => {
        const response = await fetch(`${ATLAS_DATA}/volume-states.json`);
        return response.ok
          ? statesByVolumeKey(
              (await response.json()) as Record<string, string[]>,
            )
          : null;
      })();
      const states = (await volumeStates.current)?.get(volumeKey(item)) ?? [];
      for (const state of states) {
        const byPlace = await loadStateVolumes(state);
        const place = byPlace ? placeOfVolume(byPlace, item) : null;
        if (place) return place;
      }
      return null;
    },
    [loadStateVolumes],
  );

  const onViewChange = useCallback(
    (bounds: Bounds, nextZoom: number) => {
      setZoom(nextZoom);
      // Half a level early, so the footprints are there when they appear.
      if (!footprintIndex || nextZoom < FOOTPRINT_ZOOM - 0.5) return;
      const wanted = townsInView(footprintIndex.towns, bounds).slice(
        0,
        MAX_TOWNS_PER_VIEW,
      );
      void inParallel(wanted, FETCH_WIDTH, loadTown);
    },
    [footprintIndex, loadTown],
  );

  /** Frame a volume: its shape, else the edition's, else the town's dot. */
  const frame = useCallback(
    (group: VolumeGroup, item: string | null, data: TownData) => {
      const shape =
        (group.kind === 'series'
          ? data.footprints?.series[group.index]?.footprint
          : null) ?? (item ? data.footprints?.volumes[item]?.footprint : null);
      if (shape) {
        setTarget({ bounds: multiPolygonBounds(shape), key: Date.now() });
        return;
      }
      const dot = placeById.get(group.place);
      if (dot) {
        setTarget({
          center: [dot.lon, dot.lat],
          zoom: TOWN_ZOOM,
          key: Date.now(),
        });
      }
    },
    [placeById],
  );

  /**
   * Select a volume of a town, on an edition: the given one, else its newest
   * covering `point`, else its newest placed. `fit` frames it, for a
   * selection made from somewhere other than the map.
   */
  const selectGroup = useCallback(
    async (options: {
      place: string;
      group: (groups: VolumeGroup[], data: TownData) => VolumeGroup | null;
      item?: string;
      point: [number, number] | null;
      fit: boolean;
    }) => {
      const data = await loadTownData(options.place);
      const groups = volumeGroups(options.place, data.volumes, data.footprints);
      const group = options.group(groups, data);
      if (!group) return;
      const item =
        options.item ??
        defaultEdition(
          group.editions,
          data.footprints?.volumes ?? {},
          options.point,
        )?.item ??
        null;
      setSelection({
        place: options.place,
        group: group.index,
        item,
        point: options.point,
      });
      setSelectedPage(null);
      if (options.fit) frame(group, item, data);
      setUrlItem(item);
    },
    [loadTownData, frame],
  );

  // A town picked by name or dot: the volume at its own point, else the one
  // holding its newest placed edition, else its first.
  const selectPlace = useCallback(
    (place: Place) =>
      selectGroup({
        place: place.id,
        point: null,
        fit: true,
        group: (groups, data) => {
          const town = data.footprints;
          const here = town
            ? seriesAt(place.lon, place.lat, new Map([[place.id, town]]))[0]
            : undefined;
          if (here) return groups[here.index] ?? null;
          const newest = defaultEdition(
            groups
              .flatMap((g) => g.editions)
              .sort((a, b) => (b.year ?? -1) - (a.year ?? -1)),
            town?.volumes ?? {},
            null,
          );
          const holding = newest ? groupOfItem(groups, newest.item) : null;
          return holding ?? groups[0] ?? null;
        },
      }),
    [selectGroup],
  );

  const onPickLocation = useCallback(
    (lng: number, lat: number) => {
      const hit = seriesAt(lng, lat, towns)[0];
      if (!hit) {
        setSelectedPage(null);
        return;
      }
      if (
        selection &&
        hit.place === selection.place &&
        hit.index === selection.group
      ) {
        setSelection({ ...selection, point: [lng, lat] });
        setSelectedPage(null);
        return;
      }
      void selectGroup({
        place: hit.place,
        point: [lng, lat],
        fit: false,
        group: (groups) => groups[hit.index] ?? null,
      });
    },
    [towns, selection, selectGroup],
  );

  // Open the edition a shared URL names, once the indexes are in.
  const openedFromUrl = useRef(false);
  useEffect(() => {
    if (openedFromUrl.current || !index || !footprintIndex) return;
    openedFromUrl.current = true;
    const item = itemFromUrl();
    if (!item) return;
    void (async () => {
      const place = await resolveVolumePlace(item);
      if (!place) return;
      await selectGroup({
        place,
        item,
        point: null,
        fit: true,
        group: (groups) => groupOfItem(groups, item),
      });
    })();
  }, [index, footprintIndex, selectGroup, resolveVolumePlace]);

  const placeId = selection?.place ?? null;

  // The town's catalog, for its volumes and their editions.
  useEffect(() => {
    if (!placeId) {
      setTownVolumes(null);
      return;
    }
    let canceled = false;
    setTownVolumes(null);
    void loadStateVolumes(stateSlug(placeId)).then((byPlace) => {
      if (!canceled) setTownVolumes(byPlace?.[placeId] ?? []);
    });
    return () => {
      canceled = true;
    };
  }, [placeId, loadStateVolumes]);

  const town = placeId ? (towns.get(placeId) ?? null) : null;
  const groups = useMemo(() => {
    if (!placeId || !townVolumes) return null;
    // A town with footprints waits for its file, or its groups would shuffle.
    if (townsWithFootprints.has(placeId) && !town) return null;
    return volumeGroups(placeId, townVolumes, town);
  }, [placeId, townVolumes, town, townsWithFootprints]);
  const group = (selection && groups?.[selection.group]) || null;
  const edition = useMemo(
    () =>
      group?.editions.find((volume) => volume.item === selection?.item) ?? null,
    [group, selection],
  );
  const editions = useMemo(
    () => (group ? editionOptions(group.editions, town?.volumes ?? {}) : []),
    [group, town],
  );

  // The selected edition's annotation: the only imagery drawn.
  const run = footprintIndex?.run ?? null;
  useEffect(() => {
    setResult(null);
    if (!run || !edition) return;
    let canceled = false;
    void loadVolume(edition, run, imageSource).then((loaded) => {
      if (!canceled) setResult(loaded);
    });
    return () => {
      canceled = true;
    };
  }, [run, edition, imageSource]);

  // Its item record: the sheets the run left unplaced, and the sheet numbers
  // the item viewer is linked by.
  useEffect(() => {
    setRecord(undefined);
    if (!edition?.state) {
      setRecord(null);
      return;
    }
    let canceled = false;
    void fetchItemRecord(edition.item, DATA).then((fetched) => {
      if (!canceled) setRecord(fetched);
    });
    return () => {
      canceled = true;
    };
  }, [edition]);

  // The town's other volumes that cover the spot picked.
  const alsoHere = useMemo(() => {
    if (!selection?.point || !town || !groups) return [];
    return seriesAt(
      selection.point[0],
      selection.point[1],
      new Map([[selection.place, town]]),
    )
      .filter((ref) => ref.index !== selection.group)
      .map((ref) => groups[ref.index])
      .filter((g): g is VolumeGroup => g !== undefined);
  }, [selection, town, groups]);

  const isSeries = group?.kind === 'series';
  const selectedRef = useMemo(
    () =>
      placeId && selection && isSeries
        ? { place: placeId, index: selection.group }
        : null,
    [placeId, selection?.group, isSeries],
  );
  const footprints = useMemo(
    () => footprintFeatures(towns, selectedRef),
    [towns, selectedRef],
  );
  const selectedFootprint =
    edition && town ? (town.volumes[edition.item]?.footprint ?? null) : null;

  const close = useCallback(() => {
    setSelection(null);
    setResult(null);
    setSelectedPage(null);
    setUrlItem(null);
  }, []);

  const places = index?.places ?? [];
  const panelPlace = placeId ? (placeById.get(placeId) ?? null) : null;
  const loaded = result && isLoaded(result) ? result : null;

  return (
    <div className="atlas">
      <div className="atlas-bar">
        <span className="atlas-title">Sanborn atlas</span>
        <PlaceSearch
          places={places}
          onSelect={(place) => void selectPlace(place)}
        />
        <label className="atlas-size-by">
          Sheets from
          <select
            value={imageSource}
            onChange={(event) =>
              setImageSource(event.target.value as ImageSource)
            }
          >
            <option value="chronoscope">Chronoscope</option>
            <option value="loc">loc.gov</option>
          </select>
        </label>
        <div
          className="atlas-opacity"
          title="Sheet opacity. Press p to cycle 100/50/0%; hold the space bar to hide the sheets."
        >
          <input
            type="range"
            id="atlas-opacity-slider"
            min={0}
            max={100}
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
          <label htmlFor="atlas-opacity-slider">Opacity (p)</label>
        </div>
        <label className="atlas-size-by">
          Dot size
          <select
            value={sizeBy}
            onChange={(event) =>
              setSizeBy(event.target.value as 'sheets' | 'volumes')
            }
          >
            <option value="sheets">by sheets</option>
            <option value="volumes">by volumes</option>
          </select>
        </label>
        <span className="atlas-bar-note">
          {indexError
            ? indexError
            : index && run
              ? `${places.length.toLocaleString()} towns · mapsnap ${run.version}`
              : 'loading the index…'}
        </span>
      </div>
      <div className="atlas-body">
        <AtlasMap
          places={places}
          townsWithShapes={townsWithShapes}
          sizeBy={sizeBy}
          footprints={footprints}
          hasSelection={selectedRef !== null}
          selectedFootprint={selectedFootprint}
          loaded={loaded}
          selectedPage={selectedPage}
          opacity={peeking ? 0 : opacity / 100}
          target={target}
          onSelectPlace={(place) => void selectPlace(place)}
          onPickLocation={onPickLocation}
          onSelectPage={(page, point) => {
            setSelectedPage(page);
            if (point) {
              setSelection((current) => current && { ...current, point });
            }
          }}
          onViewChange={onViewChange}
        />
        {!panelPlace && zoom >= FOOTPRINT_ZOOM && (
          <div className="atlas-hint">
            Click a shaded volume to see its editions, or a ringed dot for a
            town with no placed sheets.
          </div>
        )}
        {panelPlace && (
          <VolumePanel
            place={panelPlace}
            groups={groups}
            group={group}
            editions={editions}
            edition={edition}
            result={edition ? result : null}
            record={record}
            alsoHere={alsoHere}
            onSelectGroup={(next, { atSpot }) =>
              void selectGroup({
                place: next.place,
                // One of the spot's other volumes stays at the spot; one
                // picked from the town's list is framed whole.
                point: atSpot ? (selection?.point ?? null) : null,
                fit: !atSpot,
                group: (all) => all[next.index] ?? null,
              })
            }
            onSelectEdition={(item) => {
              if (!selection) return;
              setSelection({ ...selection, item });
              setSelectedPage(null);
              setUrlItem(item);
            }}
            selectedPage={selectedPage}
            onClose={close}
          />
        )}
      </div>
    </div>
  );
}
