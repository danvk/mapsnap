import type { IndexedDetection } from '../detections';
import type { AdjacencyData, AdjacencyPage } from '../types';
import { DetectionCanvas } from './DetectionCanvas';

interface AdjacencyTableProps {
  adjacency: AdjacencyData;
  /** Page stem from the dropped image's filename, e.g. "p49"; null until an image is dropped. */
  imageStem: string | null;
  /** The page's detections converted to Detection shape, parallel to the page's detection list. */
  detections: IndexedDetection[];
  selectedIndices: Set<number>;
  onSelect: (index: number) => void;
  image: HTMLImageElement | null;
  jsonWidth: number;
  jsonHeight: number;
}

/** The page's mutual neighbors, each with the edge its claim was printed on (or "?"). */
function neighborSummary(
  adjacency: AdjacencyData,
  stem: string,
  page: AdjacencyPage,
): { stem: string; edge: string }[] {
  const neighbors = adjacency.adjacency
    .filter(([a, b]) => a === stem || b === stem)
    .map(([a, b]) => (a === stem ? b : a));
  return neighbors.map((neighborStem) => {
    const number = adjacency.pages[neighborStem]?.number;
    const claim = page.detections.find((d) => d.claim && d.number === number);
    return { stem: neighborStem, edge: claim?.edge ?? '?' };
  });
}

/**
 * Side panel for adjacency mode: the page's mutual neighbors (with the edge
 * each was printed on) above a table of its page-number detections, claims
 * first. Clicking a row highlights that detection on the image.
 */
export function AdjacencyTable(props: AdjacencyTableProps) {
  const {
    adjacency,
    imageStem,
    detections,
    selectedIndices,
    onSelect,
    image,
    jsonWidth,
    jsonHeight,
  } = props;

  if (!imageStem) {
    return (
      <div id="detections-panel">
        <p>
          Drop this page's image to select a page — its filename (e.g.
          "p49.jpg") identifies the page in adjacency.json.
        </p>
      </div>
    );
  }
  const page = adjacency.pages[imageStem];
  if (!page) {
    return (
      <div id="detections-panel">
        <p>
          Page "{imageStem}" is not in this adjacency.json (
          {Object.keys(adjacency.pages).length} pages).
        </p>
      </div>
    );
  }

  const neighbors = neighborSummary(adjacency, imageStem, page);
  // Mutual neighbors first (the information this view exists for), then other
  // claims, then the filtered-out reads; by confidence within each group.
  const rank = (indexed: IndexedDetection): number =>
    indexed.det.mutual === true ? 2 : indexed.det.mutual === false ? 1 : 0;
  const visible = detections
    .filter(({ i }) => selectedIndices.size === 0 || selectedIndices.has(i))
    .sort((a, b) => rank(b) - rank(a) || b.det.confidence - a.det.confidence);

  return (
    <div id="detections-panel">
      <div className="adjacency-summary">
        <strong>{imageStem}</strong> — mutual neighbors:{' '}
        {neighbors.length === 0
          ? 'none'
          : neighbors.map(({ stem, edge }, i) => (
              <span key={stem}>
                {i > 0 && ', '}
                {stem} <span className="adjacency-edge">({edge})</span>
              </span>
            ))}
      </div>
      <table id="detections-table">
        <thead>
          <tr>
            <th>Number</th>
            <th>Edge</th>
            <th>Height</th>
            <th>Conf</th>
            <th>Mutual</th>
            <th>Claim</th>
            <th>Image</th>
          </tr>
        </thead>
        <tbody>
          {visible.map(({ det, i }) => {
            const raw = page.detections[i];
            const classes = [
              selectedIndices.has(i) ? 'selected' : '',
              det.mutual === true ? 'mutual' : '',
              raw?.claim ? '' : 'ignored',
            ]
              .filter(Boolean)
              .join(' ');
            return (
              <tr
                key={i}
                className={classes || undefined}
                onClick={() => onSelect(i)}
              >
                <td>{raw?.number}</td>
                <td>{raw?.edge}</td>
                <td>{raw?.height}</td>
                <td>{det.confidence.toFixed(3)}</td>
                <td>{det.mutual === true ? '✓' : ''}</td>
                <td>{raw?.claim ? '✓' : ''}</td>
                <td>
                  <DetectionCanvas
                    det={det}
                    image={image}
                    jsonWidth={jsonWidth}
                    jsonHeight={jsonHeight}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
