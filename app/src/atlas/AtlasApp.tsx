/**
 * The atlas: the Sanborn collection as one map of the country.
 *
 * Open on every town the collection covers, sized by how much of it there is.
 * Pick one and the map flies in and draws that town's most recent survey --
 * every volume of the year at once, warped onto the ground by the corpus run's
 * own annotations. The year list beside it is the town's other surveys.
 *
 * Three fetches, in widening order of cost: the place index once at startup, a
 * state's volume list when a town is picked, and one annotation per volume of
 * the chosen year. Sheet tiles are pulled by Allmaps as it draws, and only for
 * what is on screen; where they come from is the "Sheets from" control, and
 * annotations.ts explains why the mirror is the default.
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
import { AtlasMap, type PageRef } from './AtlasMap';
import { PlaceSearch } from './PlaceSearch';
import { VolumePanel } from './VolumePanel';
import {
  annotationUri,
  defaultYear,
  stateSlug,
  volumesOfYear,
  type Place,
  type PlaceIndex,
  type Volume,
} from './places';

/** How many annotations to fetch at once for one town-year. */
const FETCH_WIDTH = 6;

const base = import.meta.env.BASE_URL;

export function AtlasApp() {
  const [index, setIndex] = useState<PlaceIndex | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [sizeBy, setSizeBy] = useState<'sheets' | 'volumes'>('sheets');
  // The mirror by default: loc.gov rate-limits long before a town-year's worth
  // of tiles is drawn (see annotations.ts).
  const [imageSource, setImageSource] = useState<ImageSource>('mirror');

  const [place, setPlace] = useState<Place | null>(null);
  const [volumes, setVolumes] = useState<Volume[] | null>(null);
  const [year, setYear] = useState<number | null>(null);
  const [results, setResults] = useState<(LoadedVolume | MissingVolume)[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedPage, setSelectedPage] = useState<PageRef | null>(null);

  // A state's volume list is good for every town in it, and towns in the same
  // state get clicked in runs, so the file is worth keeping.
  const stateCache = useRef(new Map<string, Record<string, Volume[]>>());

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
  }, []);

  const selectPlace = useCallback((next: Place) => {
    setPlace(next);
    setVolumes(null);
    setYear(null);
    setResults([]);
    setSelectedPage(null);
  }, []);

  // The town's volume list, from its state's file.
  useEffect(() => {
    if (!place) return;
    let cancelled = false;
    void (async () => {
      const slug = stateSlug(place.id);
      let byPlace = stateCache.current.get(slug);
      if (!byPlace) {
        const response = await fetch(`${base}atlas/volumes/${slug}.json`);
        if (!response.ok) return;
        byPlace = (await response.json()) as Record<string, Volume[]>;
        stateCache.current.set(slug, byPlace);
      }
      if (cancelled) return;
      const own = byPlace[place.id] ?? [];
      setVolumes(own);
      setYear(defaultYear(own));
    })();
    return () => {
      cancelled = true;
    };
  }, [place]);

  // The chosen year's annotations. Results are published as one batch so the
  // map draws a complete year rather than flickering through partial ones.
  useEffect(() => {
    if (!index || !volumes) return;
    const wanted = volumesOfYear(volumes, year);
    if (wanted.length === 0) {
      setResults([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    void (async () => {
      const loadedResults = await inParallel(wanted, FETCH_WIDTH, (volume) =>
        loadVolume(volume, annotationUri(volume, index), imageSource),
      );
      if (cancelled) return;
      setResults(loadedResults);
      setLoading(false);
    })();
    return () => {
      cancelled = true;
      setLoading(false);
    };
  }, [index, volumes, year, imageSource]);

  const drawn = useMemo(() => results.filter(isLoaded), [results]);

  const close = useCallback(() => {
    setPlace(null);
    setVolumes(null);
    setResults([]);
    setSelectedPage(null);
  }, []);

  const places = index?.places ?? [];

  return (
    <div className="atlas">
      <div className="atlas-bar">
        <span className="atlas-title">Sanborn atlas</span>
        <PlaceSearch places={places} onSelect={selectPlace} />
        <label className="atlas-size-by">
          Sheets from
          <select
            value={imageSource}
            onChange={(event) =>
              setImageSource(event.target.value as ImageSource)
            }
          >
            <option value="mirror">the mirror</option>
            <option value="loc">loc.gov</option>
          </select>
        </label>
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
          loaded={drawn}
          selectedPlace={place}
          selectedPage={selectedPage}
          onSelectPlace={selectPlace}
          onSelectPage={setSelectedPage}
        />
        {place && (
          <VolumePanel
            place={place}
            volumes={volumes}
            year={year}
            onSelectYear={(next) => {
              setYear(next);
              setSelectedPage(null);
            }}
            results={results}
            loading={loading}
            selectedPage={selectedPage}
            onClose={close}
          />
        )}
      </div>
    </div>
  );
}
