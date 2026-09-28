/**
 * Type-ahead over the 8,779 towns in the index.
 *
 * Plain substring ranking against "<name>, <state>", run on every keystroke
 * over an array already in memory: the whole index is smaller than one sheet
 * of a map, so there is nothing to gain by indexing it further.
 */

import { useEffect, useRef, useState } from 'react';

import { searchPlaces, type Place } from './places';

interface PlaceSearchProps {
  places: Place[];
  onSelect: (place: Place) => void;
}

export function PlaceSearch(props: PlaceSearchProps) {
  const { places, onSelect } = props;
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  const matches = open ? searchPlaces(places, query) : [];

  // A click anywhere else closes the list; without this it survives selecting
  // a town on the map and floats over the result.
  useEffect(() => {
    const onDown = (event: MouseEvent) => {
      if (!boxRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  const choose = (place: Place) => {
    setQuery(`${place.name}, ${place.state}`);
    setOpen(false);
    onSelect(place);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      setActive((current) =>
        matches.length === 0
          ? 0
          : (current + step + matches.length) % matches.length,
      );
    } else if (event.key === 'Enter') {
      const place = matches[active];
      if (place) choose(place);
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div className="place-search" ref={boxRef}>
      <input
        type="search"
        value={query}
        placeholder="Find a town…"
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
          setActive(0);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      {matches.length > 0 && (
        <ul className="place-search-results">
          {matches.map((place, index) => (
            <li key={place.id}>
              <button
                type="button"
                className={index === active ? 'is-active' : undefined}
                onMouseEnter={() => setActive(index)}
                onClick={() => choose(place)}
              >
                <span className="place-search-name">
                  {place.name}, {place.state}
                </span>
                <span className="place-search-meta">
                  {place.volumes} vol · {place.sheets.toLocaleString()} sheets
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
