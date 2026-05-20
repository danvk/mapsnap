import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './truth-georef.css';

interface Street {
  street: string;
  x: number;
  y: number;
  lat?: number;
  lon?: number;
  dir_x?: number;
  dir_y?: number;
  dir_lon?: number;
  dir_lat?: number;
  inlier?: boolean;
}

interface IntersectionPoint {
  label_a: string;
  label_b: string;
  x: number;
  y: number;
  lat: number;
  lon: number;
  inlier: boolean;
  initial?: boolean;
}

interface Detection {
  polygon: [number, number][];
  text: string;
  confidence: number;
  angle: number;
  long_side: number;
  short_side: number;
}

interface GeorefData {
  width?: number;
  height?: number;
  corners?: [
    [number, number],
    [number, number],
    [number, number],
    [number, number],
  ];
  streets?: Street[];
  intersections?: IntersectionPoint[];
}

type Corners = [
  [number, number],
  [number, number],
  [number, number],
  [number, number],
];

const img = document.querySelector('img') as HTMLImageElement;
const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
const opacitySlider = document.getElementById(
  'opacity-slider',
) as HTMLInputElement;
const opacityValue = document.getElementById('opacity-value') as HTMLElement;
const showLabelsCheckbox = document.getElementById(
  'show-labels',
) as HTMLInputElement;
const showIntersectionsCheckbox = document.getElementById(
  'show-intersections',
) as HTMLInputElement;
const showIntersectionsOnImageCheckbox = document.getElementById(
  'show-intersections-on-image',
) as HTMLInputElement;
const colorByInlierCheckbox = document.getElementById(
  'color-by-inlier',
) as HTMLInputElement;
const filterConfidenceSlider = document.getElementById(
  'filter-confidence',
) as HTMLInputElement;
const filterShortSideSlider = document.getElementById(
  'filter-short-side',
) as HTMLInputElement;
const filterLongSideSlider = document.getElementById(
  'filter-long-side',
) as HTMLInputElement;

function streetCircleColor(): maplibregl.ExpressionSpecification | string {
  return colorByInlierCheckbox.checked
    ? ([
        'case',
        ['get', 'inlier'],
        'orange',
        '#888888',
      ] as maplibregl.ExpressionSpecification)
    : '#ff0000';
}
function streetTextColor(): string {
  return colorByInlierCheckbox.checked ? 'orange' : '#ff0000';
}

let streets: Street[] = [];
let intersections: IntersectionPoint[] = [];
let detections: Detection[] = [];
let selectedDetectionIndices = new Set<number>();
let streetsMode = false;
let svg: SVGSVGElement | null = null;
let svgW = 0;
let svgH = 0;
let jsonWidth = 0;
let jsonHeight = 0;
let precomputedCorners: Corners | null = null;
let map: maplibregl.Map | null = null;
let mapReady = false;
let lastWarpedUrl = '';

/** Set the JSON coordinate space and enforce its aspect ratio on the displayed image. */
function applyJsonDimensions(width: number, height: number): void {
  jsonWidth = width;
  jsonHeight = height;
  img.style.aspectRatio = `${width} / ${height}`;
}

// Solve 3x3 linear system Ax = b using Gaussian elimination with partial pivoting.
function solveLinear3(A: number[][], b: number[]): number[] | null {
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[maxRow][col])) maxRow = row;
    }
    [m[col], m[maxRow]] = [m[maxRow], m[col]];
    if (Math.abs(m[col][col]) < 1e-12) return null;
    for (let row = col + 1; row < 3; row++) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k <= 3; k++) m[row][k] -= factor * m[col][k];
    }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    x[i] = m[i][3];
    for (let j = i + 1; j < 3; j++) x[i] -= m[i][j] * x[j];
    x[i] /= m[i][i];
  }
  return x;
}

/** Ray-casting point-in-polygon test for a convex or concave polygon. */
function pointInPolygon(
  x: number,
  y: number,
  polygon: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/** Haversine distance in miles between two lon/lat points. */
function distanceMiles(
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number,
): number {
  const R = 3958.8;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(a));
}

/**
 * Fit affine transform lon = a0 + a1*x + a2*y (and lat) from GCPs with lat/lon,
 * then map the 4 image corners to geographic coordinates.
 * Returns [nw, ne, se, sw] in [lon, lat] order, or null if fewer than 3 GCPs.
 */
function computeCorners(
  gcps: Street[],
  width: number,
  height: number,
): Corners | null {
  const valid = gcps.filter((p) => p.lat !== undefined && p.lon !== undefined);
  if (valid.length < 3) return null;

  const AtA = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const AtbLon = [0, 0, 0];
  const AtbLat = [0, 0, 0];
  for (const p of valid) {
    const row = [1, p.x, p.y];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) AtA[i][j] += row[i] * row[j];
      AtbLon[i] += row[i] * p.lon!;
      AtbLat[i] += row[i] * p.lat!;
    }
  }

  const lonCoeff = solveLinear3(AtA, AtbLon);
  const latCoeff = solveLinear3(AtA, AtbLat);
  if (!lonCoeff || !latCoeff) return null;

  const transform = (x: number, y: number): [number, number] => [
    lonCoeff[0] + lonCoeff[1] * x + lonCoeff[2] * y,
    latCoeff[0] + latCoeff[1] * x + latCoeff[2] * y,
  ];

  return [
    transform(0, 0),
    transform(width, 0),
    transform(width, height),
    transform(0, height),
  ];
}

/** Switch UI to streets-mode: hide map/controls, show detections panel. */
function enterStreetsMode(): void {
  document.getElementById('map')!.style.display = 'none';
  for (const el of document.querySelectorAll<HTMLElement>('.opacity-control')) {
    el.style.display = 'none';
  }
  for (const el of document.querySelectorAll<HTMLElement>('.image-controls')) {
    el.style.display = 'none';
  }
  document.getElementById('detection-filters')!.style.display = 'block';
  document.getElementById('detections-panel')!.style.display = 'block';
  const wrapper = img.closest<HTMLElement>('.image-wrapper');
  if (wrapper) wrapper.style.cursor = 'crosshair';
  renderDetections();
  updateDetectionsTable();
}

/** Switch back to georef mode: restore map, checkboxes, hide detection UI. */
function exitStreetsMode(): void {
  streetsMode = false;
  detections = [];
  selectedDetectionIndices = new Set();
  document.getElementById('detections-panel')!.style.display = 'none';
  document.getElementById('detection-filters')!.style.display = 'none';
  document.getElementById('map')!.style.display = '';
  for (const el of document.querySelectorAll<HTMLElement>('.opacity-control')) {
    el.style.display = '';
  }
  for (const el of document.querySelectorAll<HTMLElement>('.image-controls')) {
    el.style.display = '';
  }
  const wrapper = img.closest<HTMLElement>('.image-wrapper');
  if (wrapper) wrapper.style.cursor = '';
}

function onFilterSliderInput(
  slider: HTMLInputElement,
  valueSpanId: string,
  decimals: number,
): void {
  document.getElementById(valueSpanId)!.textContent = parseFloat(
    slider.value,
  ).toFixed(decimals);
  renderDetections();
  updateDetectionsTable();
}

filterConfidenceSlider.addEventListener('input', () =>
  onFilterSliderInput(filterConfidenceSlider, 'filter-confidence-value', 3),
);
filterShortSideSlider.addEventListener('input', () =>
  onFilterSliderInput(filterShortSideSlider, 'filter-short-side-value', 0),
);
filterLongSideSlider.addEventListener('input', () =>
  onFilterSliderInput(filterLongSideSlider, 'filter-long-side-value', 0),
);

/** Map confidence in [0, 1] to a CSS color string (red → yellow → green). */
function confidenceColor(confidence: number): string {
  const hue = Math.round(confidence * 120); // 0 = red, 120 = green
  return `hsl(${hue}, 90%, 45%)`;
}

/** Return detections passing the current filter sliders, with original indices. */
function getFilteredDetections(): { det: Detection; i: number }[] {
  const minConf = parseFloat(filterConfidenceSlider.value);
  const minShort = parseFloat(filterShortSideSlider.value);
  const minLong = parseFloat(filterLongSideSlider.value);
  return detections
    .map((det, i) => ({ det, i }))
    .filter(
      ({ det }) =>
        det.confidence >= minConf &&
        det.short_side >= minShort &&
        det.long_side >= minLong,
    );
}

/** Draw detection polygons on the SVG overlay. Selected ones are highlighted. */
function renderDetections(): void {
  if (!svg) return;
  svg.innerHTML = '';
  for (const { det, i } of getFilteredDetections()) {
    const isSelected = selectedDetectionIndices.has(i);
    const color = isSelected ? '#ff6600' : confidenceColor(det.confidence);
    const points = det.polygon
      .map(([x, y]) => toDisplay(x, y))
      .map(([dx, dy]) => `${dx},${dy}`)
      .join(' ');

    const polygonEl = document.createElementNS(
      'http://www.w3.org/2000/svg',
      'polygon',
    );
    polygonEl.setAttribute('points', points);
    polygonEl.setAttribute('fill', color);
    polygonEl.setAttribute('fill-opacity', isSelected ? '0.25' : '0.08');
    polygonEl.setAttribute('stroke', color);
    polygonEl.setAttribute('stroke-width', isSelected ? '2.5' : '1.2');
    svg.appendChild(polygonEl);

    if (isSelected) {
      const [lx, ly] = toDisplay(det.polygon[0][0], det.polygon[0][1]);
      const label = document.createElementNS(
        'http://www.w3.org/2000/svg',
        'text',
      );
      label.setAttribute('x', String(lx));
      label.setAttribute('y', String(ly - 4));
      label.setAttribute('font-size', '11');
      label.setAttribute('font-family', 'sans-serif');
      label.setAttribute('fill', color);
      label.setAttribute('stroke', 'white');
      label.setAttribute('stroke-width', '2');
      label.setAttribute('paint-order', 'stroke');
      label.textContent = det.text;
      svg.appendChild(label);
    }
  }
}

/**
 * Draw the rotated image patch for a detection into a canvas.
 * Rotation is derived from the direction of the polygon's longest side so that
 * diagonally-oriented text is also rendered horizontally.
 */
function drawDetectionCanvas(canvas: HTMLCanvasElement, det: Detection): void {
  if (!img.naturalWidth) return;
  const cx = det.polygon.reduce((s, [x]) => s + x, 0) / 4;
  const cy = det.polygon.reduce((s, [, y]) => s + y, 0) / 4;

  // Find the direction of the longest polygon side to determine text orientation.
  let maxLen = 0;
  let longDx = 1,
    longDy = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = det.polygon[i];
    const [x2, y2] = det.polygon[(i + 1) % 4];
    const dx = x2 - x1,
      dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > maxLen) {
      maxLen = len;
      longDx = dx;
      longDy = dy;
    }
  }
  // Use det.angle (0/90/270) to disambiguate the 180° ambiguity of the long-side direction.
  // 90° and 270° both indicate vertical text; folding to the smaller angle gives the same
  // reference direction (-π/2) for both, so the disambiguation picks the correct half.
  const rawAngle = Math.atan2(longDy, longDx);
  const foldedAngle = Math.min(det.angle, 360 - det.angle);
  const octantAngle = (-foldedAngle * Math.PI) / 180;
  let textAngle = rawAngle;
  if (Math.cos(rawAngle - octantAngle) < 0) {
    // The long side points the wrong way; flip 180°.
    textAngle = rawAngle + Math.PI;
  }
  if (det.angle == 90) {
    textAngle += Math.PI;
  }

  let scale = 40 / det.short_side;
  let cW = Math.round(det.long_side * scale);
  let cH = 40;
  if (cW > 200) {
    scale = 200 / det.long_side;
    cW = 200;
    cH = Math.round(det.short_side * scale);
  }
  canvas.width = cW;
  canvas.height = cH;

  const ctx = canvas.getContext('2d')!;
  ctx.save();
  ctx.translate(cW / 2, cH / 2);
  ctx.rotate(-textAngle);
  ctx.drawImage(
    img,
    -cx * scale,
    -cy * scale,
    img.naturalWidth * scale,
    img.naturalHeight * scale,
  );
  ctx.restore();
}

/** Rebuild the detections table, showing all rows or only selected ones. */
function updateDetectionsTable(): void {
  const tbody = document.querySelector<HTMLTableSectionElement>(
    '#detections-table tbody',
  );
  if (!tbody) return;

  const visible = getFilteredDetections()
    .filter(
      ({ i }) =>
        selectedDetectionIndices.size === 0 || selectedDetectionIndices.has(i),
    )
    .sort((a, b) => b.det.confidence - a.det.confidence);

  tbody.innerHTML = '';
  for (const [rowIdx, { det, i }] of visible.entries()) {
    const tr = document.createElement('tr');
    if (selectedDetectionIndices.has(i)) tr.classList.add('selected');

    for (const val of [
      det.angle,
      det.long_side,
      det.short_side,
      det.confidence.toFixed(3),
      det.text,
    ]) {
      const td = document.createElement('td');
      td.textContent = String(val);
      tr.appendChild(td);
    }

    const imageTd = document.createElement('td');
    if (rowIdx < 10) {
      const canvas = document.createElement('canvas');
      drawDetectionCanvas(canvas, det);
      imageTd.appendChild(canvas);
    }
    tr.appendChild(imageTd);

    tr.addEventListener('click', () => {
      selectedDetectionIndices = new Set([i]);
      renderDetections();
      updateDetectionsTable();
    });

    tbody.appendChild(tr);
  }
}

/** Handle image clicks in streets-mode: select detections at the click point. */
function handleImageClick(e: MouseEvent): void {
  if (!streetsMode) return;
  const rect = img.getBoundingClientRect();
  const displayX = e.clientX - rect.left;
  const displayY = e.clientY - rect.top;
  const imgX = (displayX * jsonWidth) / svgW;
  const imgY = (displayY * jsonHeight) / svgH;

  const hit = getFilteredDetections()
    .filter(({ det }) => pointInPolygon(imgX, imgY, det.polygon))
    .map(({ i }) => i);
  selectedDetectionIndices = new Set(hit);
  renderDetections();
  updateDetectionsTable();
}

function setupMap(): void {
  map = new maplibregl.Map({
    container: 'map',
    style: {
      version: 8,
      sources: {
        osm: {
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution:
            '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        },
      },
      layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
    },
    center: [-73.99, 40.7],
    zoom: 13,
  });

  map.on('load', () => {
    mapReady = true;
    updateWarp();
    updateStreets();
    updateIntersections();
  });
}

function updateWarp(): void {
  if (!map || !mapReady) return;
  const corners =
    precomputedCorners ?? computeCorners(streets, jsonWidth, jsonHeight);
  if (!corners) return;

  const url = img.src;
  const existing = map.getSource('warped') as
    | maplibregl.ImageSource
    | undefined;
  if (existing) {
    existing.updateImage({ url, coordinates: corners });
  } else {
    map.addSource('warped', { type: 'image', url, coordinates: corners });
    map.addLayer({
      id: 'warped',
      type: 'raster',
      source: 'warped',
      paint: { 'raster-opacity': Number(opacitySlider.value) / 100 },
    });
  }

  if (url !== lastWarpedUrl) {
    lastWarpedUrl = url;
    const lons = corners.map((c) => c[0]);
    const lats = corners.map((c) => c[1]);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const newCenterLon = (minLon + maxLon) / 2;
    const newCenterLat = (minLat + maxLat) / 2;
    const currentCenter = map.getCenter();
    const dist = distanceMiles(
      currentCenter.lng,
      currentCenter.lat,
      newCenterLon,
      newCenterLat,
    );
    map.fitBounds(
      [
        [minLon, minLat],
        [maxLon, maxLat],
      ],
      { padding: 40, maxZoom: 17, animate: dist <= 10 },
    );
  }
}

/**
 * Show street label positions and direction vectors on the map.
 * Circle + name label for each label position; line for each street direction.
 */
function updateStreets(): void {
  if (!map || !mapReady) return;

  const geo = streets.filter((s) => s.lat !== undefined && s.lon !== undefined);

  const pointsGeojson: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: geo.map((s) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates: [s.lon!, s.lat!] },
      properties: { label: s.street, inlier: s.inlier ?? true },
    })),
  };

  // Direction arrows: extend ±arrowHalf degrees from the label position.
  const arrowHalf = 0.0005;
  const linesGeojson: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: geo
      .filter((s) => s.dir_lon !== undefined && s.dir_lat !== undefined)
      .map((s) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'LineString' as const,
          coordinates: [
            [s.lon! - s.dir_lon! * arrowHalf, s.lat! - s.dir_lat! * arrowHalf],
            [s.lon! + s.dir_lon! * arrowHalf, s.lat! + s.dir_lat! * arrowHalf],
          ],
        },
        properties: {},
      })),
  };

  const existingPts = map.getSource('street-labels') as
    | maplibregl.GeoJSONSource
    | undefined;
  if (existingPts) {
    existingPts.setData(pointsGeojson);
  } else {
    map.addSource('street-labels', { type: 'geojson', data: pointsGeojson });
    map.addLayer({
      id: 'street-labels-circle',
      type: 'circle',
      source: 'street-labels',
      paint: {
        'circle-radius': 5,
        'circle-color': streetCircleColor(),
        'circle-stroke-color': '#ffffff',
        'circle-stroke-width': 1.5,
      },
    });
    map.addLayer({
      id: 'street-labels-text',
      type: 'symbol',
      source: 'street-labels',
      layout: {
        'text-field': ['get', 'label'],
        'text-font': ['Open Sans Regular'],
        'text-size': 10,
        'text-offset': [0, 1.2],
        'text-anchor': 'top',
      },
      paint: {
        'text-color': streetTextColor(),
        'text-halo-color': '#ffffff',
        'text-halo-width': 1.5,
      },
    });
  }

  const existingLines = map.getSource('street-vectors') as
    | maplibregl.GeoJSONSource
    | undefined;
  if (existingLines) {
    existingLines.setData(linesGeojson);
  } else {
    map.addSource('street-vectors', { type: 'geojson', data: linesGeojson });
    map.addLayer({
      id: 'street-vectors-line',
      type: 'line',
      source: 'street-vectors',
      paint: {
        'line-color': streetTextColor(),
        'line-width': 2,
        'line-opacity': 0.9,
      },
    });
  }

  if (map.getLayer('street-labels-circle'))
    map.setPaintProperty(
      'street-labels-circle',
      'circle-color',
      streetCircleColor(),
    );
  if (map.getLayer('street-labels-text'))
    map.setPaintProperty('street-labels-text', 'text-color', streetTextColor());
  if (map.getLayer('street-vectors-line'))
    map.setPaintProperty(
      'street-vectors-line',
      'line-color',
      streetTextColor(),
    );

  const visible = showLabelsCheckbox.checked ? 'visible' : 'none';
  for (const id of [
    'street-labels-circle',
    'street-labels-text',
    'street-vectors-line',
  ]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
  }
}

/** Show intersection GCPs (actual street crossing coordinates) on the map. */
function updateIntersections(): void {
  if (!map || !mapReady) return;

  const colorExpr = [
    'case',
    ['get', 'initial'],
    '#0080ff',
    ['get', 'inlier'],
    '#ff0000',
    '#e6b800',
  ] as maplibregl.ExpressionSpecification;

  const geojson: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: intersections.map((ix) => ({
      type: 'Feature' as const,
      geometry: { type: 'Point' as const, coordinates: [ix.lon, ix.lat] },
      properties: {
        label: `${ix.label_a}\n${ix.label_b}`,
        inlier: ix.inlier,
        initial: ix.initial ?? false,
      },
    })),
  };

  const existing = map.getSource('intersections') as
    | maplibregl.GeoJSONSource
    | undefined;
  if (existing) {
    existing.setData(geojson);
  } else {
    map.addSource('intersections', { type: 'geojson', data: geojson });
    map.addLayer({
      id: 'intersections-circle',
      type: 'circle',
      source: 'intersections',
      paint: {
        'circle-radius': 7,
        'circle-color': colorExpr,
        'circle-stroke-color': '#ffffff',
        'circle-stroke-width': 2,
      },
    });
    map.addLayer({
      id: 'intersections-text',
      type: 'symbol',
      source: 'intersections',
      layout: {
        'text-field': ['get', 'label'],
        'text-font': ['Open Sans Regular'],
        'text-size': 10,
        'text-offset': [1.4, 0],
        'text-anchor': 'left',
        'text-justify': 'left',
      },
      paint: {
        'text-color': colorExpr,
        'text-halo-color': '#ffffff',
        'text-halo-width': 2,
      },
    });
  }

  const visible = showIntersectionsCheckbox.checked ? 'visible' : 'none';
  for (const id of ['intersections-circle', 'intersections-text']) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
  }
}

async function init(): Promise<void> {
  if (!img.complete) {
    await new Promise<void>((resolve) => {
      img.addEventListener('load', () => resolve(), { once: true });
      img.addEventListener('error', () => resolve(), { once: true });
    });
  }

  setupOverlay();
  setupFileDrop();
  setupMap();

  if (img.hasAttribute('src')) {
    const jsonUrl = img.src.replace(/\.[^.]+$/, '.json');
    try {
      const resp = await fetch(jsonUrl);
      if (resp.ok) {
        const data = (await resp.json()) as GeorefData;
        streets = (data.streets ?? []).map((s) => ({ ...s }));
        intersections = (data.intersections ?? []).map((ix) => ({ ...ix }));
        precomputedCorners = data.corners ?? null;
        applyJsonDimensions(
          data.width ?? img.naturalWidth,
          data.height ?? img.naturalHeight,
        );
      } else {
        applyJsonDimensions(img.naturalWidth, img.naturalHeight);
      }
    } catch {
      applyJsonDimensions(img.naturalWidth, img.naturalHeight);
    }
  }

  syncTextarea();
  render();
}

function setupOverlay(): void {
  const wrapper = document.createElement('div');
  wrapper.className = 'image-wrapper';
  img.replaceWith(wrapper);
  wrapper.appendChild(img);

  svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.style.position = 'absolute';
  svg.style.top = '0';
  svg.style.left = '0';
  svg.style.pointerEvents = 'none';
  wrapper.appendChild(svg);

  function updateSize(): void {
    svgW = img.offsetWidth;
    svgH = img.offsetHeight;
    svg!.setAttribute('width', String(svgW));
    svg!.setAttribute('height', String(svgH));
    render();
  }

  new ResizeObserver(updateSize).observe(img);
  updateSize();

  wrapper.addEventListener('click', handleImageClick);
}

function setupFileDrop(): void {
  let prevObjectUrl: string | null = null;

  // Drop an image file onto the image-column (covers both placeholder and image).
  const imageColumn = img.parentElement!.parentElement!;
  const dropPlaceholder = imageColumn.querySelector(
    '.drop-placeholder',
  ) as HTMLElement;
  imageColumn.addEventListener('dragover', (e) => {
    if ([...(e as DragEvent).dataTransfer!.types].includes('Files')) {
      e.preventDefault();
      dropPlaceholder.classList.add('drag-over');
    }
  });
  imageColumn.addEventListener('dragleave', () => {
    dropPlaceholder.classList.remove('drag-over');
  });
  async function processJsonFile(file: File): Promise<void> {
    const text = await file.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return;
    }
    if (Array.isArray(parsed)) {
      // streets.json: array of detection objects from detect_text.py
      detections = (parsed as Detection[]).filter((d) => d.confidence > 0);
      selectedDetectionIndices = new Set();
      streetsMode = true;
      textarea.value = text;
      applyJsonDimensions(
        img.naturalWidth || jsonWidth || 1,
        img.naturalHeight || jsonHeight || 1,
      );
      enterStreetsMode();
    } else {
      if (streetsMode) exitStreetsMode();
      textarea.value = text;
      textarea.dispatchEvent(new Event('input'));
    }
  }

  imageColumn.addEventListener('drop', async (e) => {
    dropPlaceholder.classList.remove('drag-over');
    const de = e as DragEvent;
    const files = [...de.dataTransfer!.files];
    const imageFile = files.find((f) => f.type.startsWith('image/'));
    const jsonFile = files.find((f) => f.name.endsWith('.json'));
    if (!imageFile && !jsonFile) return;
    de.preventDefault();

    if (imageFile) {
      if (prevObjectUrl) URL.revokeObjectURL(prevObjectUrl);
      prevObjectUrl = URL.createObjectURL(imageFile);
      img.src = prevObjectUrl;
      await new Promise<void>((resolve) => {
        img.addEventListener('load', () => resolve(), { once: true });
      });
      dropPlaceholder.style.display = 'none';
      applyJsonDimensions(img.naturalWidth, img.naturalHeight);
      syncTextarea();
    }
    if (jsonFile) {
      await processJsonFile(jsonFile);
    }
  });

  // Drop a JSON file onto the textarea to replace the georef data.
  textarea.addEventListener('dragover', (e) => {
    if ([...(e as DragEvent).dataTransfer!.types].includes('Files'))
      e.preventDefault();
  });
  textarea.addEventListener('drop', async (e) => {
    const de = e as DragEvent;
    const file = [...de.dataTransfer!.files].find((f) =>
      f.name.endsWith('.json'),
    );
    if (!file) return;
    de.preventDefault();
    await processJsonFile(file);
  });
}

/** Scale JSON coordinate space to SVG display coords. */
function toDisplay(nx: number, ny: number): [number, number] {
  return [(nx * svgW) / jsonWidth, (ny * svgH) / jsonHeight];
}

function render(): void {
  if (streetsMode) {
    renderDetections();
    return;
  }
  if (!svg) return;
  svg.innerHTML = '';

  // Street label dots with direction arrows. Outliers grey, inliers orange.
  // Render outliers first so inliers appear on top when they overlap.
  const sortedStreets = [...streets].sort(
    (a, b) => (a.inlier ? 1 : 0) - (b.inlier ? 1 : 0),
  );
  for (const st of sortedStreets) {
    const [cx, cy] = toDisplay(st.x, st.y);
    const color = !colorByInlierCheckbox.checked
      ? '#ff0000'
      : st.inlier !== false
        ? 'orange'
        : '#888888';
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

    const circle = document.createElementNS(
      'http://www.w3.org/2000/svg',
      'circle',
    );
    circle.setAttribute('cx', String(cx));
    circle.setAttribute('cy', String(cy));
    circle.setAttribute('r', '8');
    circle.setAttribute('fill', color);
    circle.setAttribute('fill-opacity', '0.7');
    circle.setAttribute('stroke', 'white');
    circle.setAttribute('stroke-width', '2');
    g.appendChild(circle);

    if (st.dir_x !== undefined && st.dir_y !== undefined) {
      const arrowLen = 40;
      const dx = (st.dir_x * svgW) / jsonWidth;
      const dy = (st.dir_y * svgH) / jsonHeight;
      const len = Math.sqrt(dx * dx + dy * dy);
      const ndx = (dx / len) * arrowLen;
      const ndy = (dy / len) * arrowLen;

      for (const sign of [1, -1] as const) {
        const line = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'line',
        );
        line.setAttribute('x1', String(cx));
        line.setAttribute('y1', String(cy));
        line.setAttribute('x2', String(cx + sign * ndx));
        line.setAttribute('y2', String(cy + sign * ndy));
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', '2');
        line.setAttribute('stroke-opacity', '0.9');
        g.appendChild(line);
      }
    }

    const label = document.createElementNS(
      'http://www.w3.org/2000/svg',
      'text',
    );
    label.setAttribute('x', String(cx + 12));
    label.setAttribute('y', String(cy - 4));
    label.setAttribute('font-size', '13');
    label.setAttribute('font-family', 'sans-serif');
    label.setAttribute('font-weight', 'bold');
    label.setAttribute('fill', color);
    label.setAttribute('stroke', 'white');
    label.setAttribute('stroke-width', '3');
    label.setAttribute('paint-order', 'stroke');
    label.textContent = st.street;
    g.appendChild(label);

    svg.appendChild(g);
  }

  // Intersection GCPs: blue = initial seed, red = inlier, yellow = outlier.
  // Render in ascending priority so initial seeds appear on top.
  const ixPriority = (ix: IntersectionPoint) =>
    ix.initial ? 2 : ix.inlier ? 1 : 0;
  if (showIntersectionsOnImageCheckbox.checked) {
    for (const ix of [...intersections].sort(
      (a, b) => ixPriority(a) - ixPriority(b),
    )) {
      const [cx, cy] = toDisplay(ix.x, ix.y);
      const color = ix.initial ? '#0080ff' : ix.inlier ? '#ff0000' : '#e6b800';
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

      const circle = document.createElementNS(
        'http://www.w3.org/2000/svg',
        'circle',
      );
      circle.setAttribute('cx', String(cx));
      circle.setAttribute('cy', String(cy));
      circle.setAttribute('r', '6');
      circle.setAttribute('fill', color);
      circle.setAttribute('fill-opacity', '0.85');
      circle.setAttribute('stroke', 'white');
      circle.setAttribute('stroke-width', '2');
      g.appendChild(circle);

      const label = document.createElementNS(
        'http://www.w3.org/2000/svg',
        'text',
      );
      label.setAttribute('x', String(cx + 10));
      label.setAttribute('y', String(cy - 2));
      label.setAttribute('font-size', '10');
      label.setAttribute('font-family', 'sans-serif');
      label.setAttribute('fill', color);
      label.setAttribute('stroke', 'white');
      label.setAttribute('stroke-width', '2');
      label.setAttribute('paint-order', 'stroke');
      for (const [i, name] of [ix.label_a, ix.label_b].entries()) {
        const tspan = document.createElementNS(
          'http://www.w3.org/2000/svg',
          'tspan',
        );
        tspan.setAttribute('x', String(cx + 10));
        tspan.setAttribute('dy', i === 0 ? '0' : '12');
        tspan.textContent = name;
        label.appendChild(tspan);
      }
      g.appendChild(label);

      svg.appendChild(g);
    }
  }
}

/** Write current state back to the textarea (read-only display). */
function syncTextarea(): void {
  textarea.value = JSON.stringify(
    {
      width: jsonWidth,
      height: jsonHeight,
      corners: precomputedCorners,
      streets,
      intersections,
    },
    null,
    2,
  );
}

textarea.addEventListener('input', () => {
  if (streetsMode) return;
  try {
    const data = JSON.parse(textarea.value) as GeorefData;
    streets = (data.streets ?? []).map((s) => ({ ...s }));
    intersections = (data.intersections ?? []).map((ix) => ({ ...ix }));
    precomputedCorners = data.corners ?? null;
    if (data.width && data.height) applyJsonDimensions(data.width, data.height);
    render();
    updateWarp();
    updateStreets();
    updateIntersections();
  } catch {
    // invalid JSON mid-edit, skip re-render
  }
});

opacitySlider.addEventListener('input', () => {
  const opacity = Number(opacitySlider.value) / 100;
  opacityValue.textContent = `${opacitySlider.value}%`;
  if (map && mapReady && map.getLayer('warped')) {
    map.setPaintProperty('warped', 'raster-opacity', opacity);
  }
});

showLabelsCheckbox.addEventListener('change', () => {
  if (!map || !mapReady) return;
  const visible = showLabelsCheckbox.checked ? 'visible' : 'none';
  for (const id of [
    'street-labels-circle',
    'street-labels-text',
    'street-vectors-line',
  ]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
  }
});

showIntersectionsOnImageCheckbox.addEventListener('change', () => {
  render();
});

colorByInlierCheckbox.addEventListener('change', () => {
  render();
  updateStreets();
});

showIntersectionsCheckbox.addEventListener('change', () => {
  if (!map || !mapReady) return;
  const visible = showIntersectionsCheckbox.checked ? 'visible' : 'none';
  for (const id of ['intersections-circle', 'intersections-text']) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible);
  }
});

void init();
