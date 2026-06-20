import type {
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
      width: number;
      height: number;
    };

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

/**
 * Classify dropped JSON text as a streets detection list, panels sidecar, or
 * georef data.
 *
 * streets.json comes either as a bare array of detections (old format) or as an
 * object with a `streets` array plus image metadata (new format). panels.json is
 * an object with a `panels` array of polygon rings plus image metadata. Anything
 * else is treated as georef data. Returns `{ kind: 'invalid' }` if the text is
 * not JSON.
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

  if (!Array.isArray(parsed) && isPanelsFormat(parsed)) {
    return {
      kind: 'panels',
      text,
      panels: parsed.panels,
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
