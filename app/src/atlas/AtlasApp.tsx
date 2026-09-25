/**
 * The atlas: the Sanborn collection as one map of the country.
 *
 * Zoomed out, every town the collection covers is a dot, sized by how much of
 * it there is. Zoomed in, the dots become volume footprints -- where each
 * digitized volume's sheets lie, drawn as each town's newest coverage -- and
 * the volume is the unit: a click picks the newest volume covering that spot
 * and draws its sheets, and the panel offers the same spot in the town's other
 * years. One volume is drawn at a time, which keeps what is fetched and warped
 * to one volume's sheets and nudges viewing toward the zoom levels where the
 * scans look good.
 *
 * Fetches, in widening order of cost: the place index and the footprint index
 * once at startup; a town's footprint file when its bounds come into view; a
 * state's volume list when one of its volumes is selected; and that volume's
 * annotation. Sheet tiles are pulled by Allmaps as it draws, and only for what
 * is on screen; where they come from is the "Sheets from" control, and
 * annotations.ts explains why the CDN is the default.
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
  footprintFeatures,
  multiPolygonBounds,
  newestVolumeAt,
  townsInView,
  yearOptions,
  type FootprintIndex,
  type TownFootprints,
  type VolumeRef,
} from './footprints';
import { nextOpacity } from './opacity';
import { PlaceSearch } from './PlaceSearch';
import { VolumePanel } from './VolumePanel';
import {
  annotationUri,
  stateSlug,
  type Place,
  type PlaceIndex,
  type Volume,
} from './places';

/** How many town footprint files to fetch at once. */
const FETCH_WIDTH = 6;
/** At most this many towns' footprints are fetched for one view. */
const MAX_TOWNS_PER_VIEW = 80;

const base = import.meta.env.BASE_URL;

/** The selected volume, and the spot its years are looked up for. */
interface Selection extends VolumeRef {
  point: [number, number];
}

// The volume a page URL names, if any: ?volume=<item>&place=<state>/<town>.
function volumeFromUrl(): VolumeRef | null {
  const params = new URLSearchParams(window.location.search);
  const item = params.get('volume');
  const place = params.get('place');
  return item && place ? { item, place } : null;
}

export function AtlasApp() {
  const [index, setIndex] = useState<PlaceIndex | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [footprintIndex, setFootprintIndex] = useState<FootprintIndex | null>(
    null,
  );
  const [sizeBy, setSizeBy] = useState<'sheets' | 'volumes'>('sheets');
  // The CDN by default: loc.gov rate-limits long before a town-year's worth of
  // tiles is drawn (see annotations.ts).
  const [imageSource, setImageSource] = useState<ImageSource>('cdn');
  // Sheet opacity in percent, as in the volume viewer: a slider, and `p` to
  // step through 100/50/0 so the map underneath can be checked at a keypress.
  const [opacity, setOpacity] = useState(100);
  useEffect(() => {
    function onKeydown(event: KeyboardEvent): void {
      if (event.key !== 'p' || isTypingTarget(event.target)) return;
      setOpacity(nextOpacity);
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, []);

  const [towns, setTowns] = useState<ReadonlyMap<string, TownFootprints>>(
    () => new Map(),
  );
  const [selection, setSelection] = useState<Selection | null>(null);
  // A town picked by name or dot that has no volume to select.
  const [bareTown, setBareTown] = useState<Place | null>(null);
  const [townVolumes, setTownVolumes] = useState<Volume[] | null>(null);
  const [result, setResult] = useState<LoadedVolume | MissingVolume | null>(
    null,
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

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch(`${base}atlas/places.json`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setIndex((await response.json()) as PlaceIndex);
      } catch (error) {
        setIndexError(
          `${String(error)} — run scripts/atlas/build_places.py to generate it`,
        );
      }
    })();
    void (async () => {
      const response = await fetch(`${base}atlas/footprints/index.json`);
      if (response.ok) {
        setFootprintIndex((await response.json()) as FootprintIndex);
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
        const response = await fetch(`${base}atlas/footprints/${place}.json`);
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
        const response = await fetch(`${base}atlas/volumes/${slug}.json`);
        return response.ok
          ? ((await response.json()) as Record<string, Volume[]>)
          : null;
      })();
      stateCache.current.set(slug, pending);
    }
    return pending;
  }, []);

  const onViewChange = useCallback(
    (bounds: [number, number, number, number], nextZoom: number) => {
      setZoom(nextZoom);
      // Half a level early, so the footprints are there when they appear.
      if (!footprintIndex || nextZoom < FOOTPRINT_ZOOM - 0.5) return;
      const wanted = townsInView(footprintIndex, bounds).slice(
        0,
        MAX_TOWNS_PER_VIEW,
      );
      void inParallel(wanted, FETCH_WIDTH, loadTown);
    },
    [footprintIndex, loadTown],
  );

  /**
   * Select a volume, and the spot to find it in other years at: the clicked
   * point, or else a point inside its footprint. `fit` frames it, for a
   * selection made from somewhere other than the map.
   */
  const selectVolume = useCallback(
    async (ref: VolumeRef, point: [number, number] | null, fit: boolean) => {
      const town = await loadTown(ref.place);
      const footprint = town?.[ref.item];
      const spot = point ?? footprint?.anchor ?? null;
      if (!spot) return;
      setBareTown(null);
      setSelection({ ...ref, point: spot });
      setSelectedPage(null);
      if (fit && footprint) {
        setTarget({
          bounds: multiPolygonBounds(footprint.footprint),
          key: Date.now(),
        });
      }
      window.history.replaceState(
        null,
        '',
        `?volume=${encodeURIComponent(ref.item)}&place=${encodeURIComponent(ref.place)}`,
      );
    },
    [loadTown],
  );

  // A town picked by name or dot: its newest volume at the town's own point,
  // else its newest volume anywhere; a town with none is shown bare.
  const selectPlace = useCallback(
    async (place: Place) => {
      const town = footprintIndex?.[place.id] ? await loadTown(place.id) : null;
      const entries = Object.entries(town ?? {});
      if (entries.length > 0) {
        const here = newestVolumeAt(
          place.lon,
          place.lat,
          new Map([[place.id, town!]]),
        );
        const newest = entries.reduce((a, b) =>
          (b[1].year ?? -Infinity) > (a[1].year ?? -Infinity) ? b : a,
        );
        await selectVolume(
          here ?? { item: newest[0], place: place.id },
          here ? [place.lon, place.lat] : null,
          true,
        );
        return;
      }
      setSelection(null);
      setResult(null);
      setSelectedPage(null);
      setBareTown(place);
      setTarget({ center: [place.lon, place.lat], zoom: 13, key: Date.now() });
    },
    [footprintIndex, loadTown, selectVolume],
  );

  const onPickLocation = useCallback(
    (lng: number, lat: number) => {
      const hit = newestVolumeAt(lng, lat, towns);
      if (hit) void selectVolume(hit, [lng, lat], false);
    },
    [towns, selectVolume],
  );

  // Open the volume a shared URL names, once the indexes are in.
  const openedFromUrl = useRef(false);
  useEffect(() => {
    if (openedFromUrl.current || !index || !footprintIndex) return;
    openedFromUrl.current = true;
    const fromUrl = volumeFromUrl();
    if (fromUrl) void selectVolume(fromUrl, null, true);
  }, [index, footprintIndex, selectVolume]);

  const placeId = selection?.place ?? bareTown?.id ?? null;

  // The town's catalogue, for the year buttons and the volume's own entry.
  useEffect(() => {
    if (!placeId) {
      setTownVolumes(null);
      return;
    }
    let cancelled = false;
    setTownVolumes(null);
    void loadStateVolumes(stateSlug(placeId)).then((byPlace) => {
      if (!cancelled) setTownVolumes(byPlace?.[placeId] ?? []);
    });
    return () => {
      cancelled = true;
    };
  }, [placeId, loadStateVolumes]);

  const volume = useMemo(
    () => townVolumes?.find((entry) => entry.item === selection?.item) ?? null,
    [townVolumes, selection],
  );

  // The selected volume's annotation: the only imagery drawn.
  useEffect(() => {
    setResult(null);
    if (!index || !volume) return;
    let cancelled = false;
    void loadVolume(volume, annotationUri(volume, index), imageSource).then(
      (loaded) => {
        if (!cancelled) setResult(loaded);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [index, volume, imageSource]);

  const years = useMemo(() => {
    const footprints = selection ? towns.get(selection.place) : undefined;
    if (!selection || !townVolumes || !footprints) return [];
    return yearOptions(
      selection.point,
      selection.item,
      townVolumes,
      footprints,
    );
  }, [selection, townVolumes, towns]);

  const footprints = useMemo(
    () => footprintFeatures(towns, selection?.item ?? null),
    [towns, selection],
  );
  const selectedFootprint = selection
    ? (towns.get(selection.place)?.[selection.item]?.footprint ?? null)
    : null;

  const close = useCallback(() => {
    setSelection(null);
    setBareTown(null);
    setResult(null);
    setSelectedPage(null);
    window.history.replaceState(null, '', window.location.pathname);
  }, []);

  const places = index?.places ?? [];
  const panelPlace = placeId
    ? (places.find((entry) => entry.id === placeId) ?? null)
    : null;
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
            <option value="cdn">Chronoscope</option>
            <option value="loc">loc.gov</option>
          </select>
        </label>
        <div
          className="atlas-opacity"
          title="Sheet opacity. Press p to cycle 100/50/0%."
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
            : index
              ? `${places.length.toLocaleString()} towns · run ${index.runTag}`
              : 'loading the index…'}
        </span>
      </div>
      <div className="atlas-body">
        <AtlasMap
          places={places}
          sizeBy={sizeBy}
          footprints={footprints}
          selectedFootprint={selectedFootprint}
          loaded={loaded}
          selectedPage={selectedPage}
          opacity={opacity / 100}
          target={target}
          onSelectPlace={(place) => void selectPlace(place)}
          onPickLocation={onPickLocation}
          onSelectPage={setSelectedPage}
          onViewChange={onViewChange}
        />
        {!panelPlace && zoom >= FOOTPRINT_ZOOM && (
          <div className="atlas-hint">
            Click a shaded area to see the newest map of that spot.
          </div>
        )}
        {panelPlace && (
          <VolumePanel
            place={panelPlace}
            townVolumes={townVolumes}
            volume={volume}
            result={volume ? result : null}
            years={years}
            onSelectItem={(item) => {
              if (selection) {
                void selectVolume(
                  { item, place: selection.place },
                  selection.point,
                  false,
                );
              }
            }}
            selectedPage={selectedPage}
            onClose={close}
          />
        )}
      </div>
    </div>
  );
}
