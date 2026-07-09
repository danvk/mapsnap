import type {
  AdjacencyData,
  Detection,
  PanelPolygon,
  PanelsJsonData,
  StreetsJsonData,
} from './types';

/** Result of classifying a dropped JSON file. */
export type ParsedJson =
  | { kind: 'invalid' }
  | { kind: 'georef'; text: string }
  | {
      kind: 'streets';
      text: string;
      detections: Detection[];
      width: number;
      height: number;
    }
  | {
      kind: 'panels';
      text: string;
      panels: PanelPolygon[];
      labels?: string[];
      width: number;
      height: number;
    }
  | { kind: 'adjacency'; data: AdjacencyData };

/**
 * Page stem from an image file name or URL: the basename with every extension
 * stripped, so "data/vol/p50n.2048px.jpg" -> "p50n". Mirrors Python's
 * `image_stem`, letting a dropped page image identify its page in a
 * volume-level file like adjacency.json.
 */
export function pageStem(fileName: string): string {
  const base = fileName.split(/[\\/]/).pop() ?? fileName;
  return base.split('.')[0];
}

/** Fallback image dimensions used when an old-format streets.json omits them. */
export interface FallbackDimensions {
  width: number;
  height: number;
}

// Whether a parsed object is a new-format streets.json (wrapped detection list).
function isNewStreetsFormat(parsed: unknown): parsed is StreetsJsonData {
  return (
    typeof parsed === 'object' &&
    parsed !== null &&
    'streets' in parsed &&
    Array.isArray((parsed as StreetsJsonData).streets) &&
    ((parsed as StreetsJsonData).streets[0] as Partial<Detection> | undefined)
      ?.confidence !== undefined
  );
}

// Whether a parsed object is a panels.json sidecar (polygon list + image metadata).
function isPanelsFormat(parsed: unknown): parsed is PanelsJsonData {
  return (
    typeof parsed === 'object' &&
    parsed !== null &&
    'panels' in parsed &&
    Array.isArray((parsed as PanelsJsonData).panels)
  );
}

// Whether a parsed object is a volume adjacency.json (per-page detections + edge list).
function isAdjacencyFormat(parsed: unknown): parsed is AdjacencyData {
  return (
    typeof parsed === 'object' &&
    parsed !== null &&
    'pages' in parsed &&
    'adjacency' in parsed &&
    typeof (parsed as AdjacencyData).pages === 'object' &&
    Array.isArray((parsed as AdjacencyData).adjacency)
  );
}

/**
 * Classify dropped JSON text as a streets detection list, panels sidecar,
 * volume adjacency data, or georef data.
 *
 * streets.json comes either as a bare array of detections (old format) or as an
 * object with a `streets` array plus image metadata (new format). panels.json is
 * an object with a `panels` array of polygon rings plus image metadata.
 * adjacency.json is an object with `pages` and `adjacency` keys. Anything else
 * is treated as georef data. Returns `{ kind: 'invalid' }` if the text is not
 * JSON.
 */
export function parseDroppedJson(
  text: string,
  fallback: FallbackDimensions,
): ParsedJson {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { kind: 'invalid' };
  }

  if (!Array.isArray(parsed) && isAdjacencyFormat(parsed)) {
    return { kind: 'adjacency', data: parsed };
  }

  if (!Array.isArray(parsed) && isPanelsFormat(parsed)) {
    return {
      kind: 'panels',
      text,
      panels: parsed.panels,
      labels: parsed.labels,
      width: parsed.width || fallback.width || 1,
      height: parsed.height || fallback.height || 1,
    };
  }

  const isOldStreetsFormat = Array.isArray(parsed);
  const isNewFormat = !isOldStreetsFormat && isNewStreetsFormat(parsed);

  if (isOldStreetsFormat || isNewFormat) {
    const rawDetections: Detection[] = isOldStreetsFormat
      ? (parsed as Detection[])
      : (parsed as StreetsJsonData).streets;
    const detections = rawDetections.filter((d) => d.confidence > 0);
    const width = isNewFormat
      ? (parsed as StreetsJsonData).width
      : fallback.width || 1;
    const height = isNewFormat
      ? (parsed as StreetsJsonData).height
      : fallback.height || 1;
    return { kind: 'streets', text, detections, width, height };
  }

  return { kind: 'georef', text };
}
