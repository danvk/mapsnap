// Print one JSON line per annotation file: {path, maps, failures, failureRates}.
// `failures` are the maps that fail as the viewer draws them (quarter size, as
// written); `failureRates` gives each map's share of RENDERINGS that fail.
//   node scripts/triangulate-masks.ts <annotation.json>...
import fs from 'node:fs';
import { parseAnnotation } from '@allmaps/annotation';
import {
  failureRates,
  triangulationFailures,
} from '../server/maskTriangulation.ts';

for (const path of process.argv.slice(2)) {
  const annotation = JSON.parse(fs.readFileSync(path, 'utf8'));
  const maps = parseAnnotation(annotation).length;
  console.log(
    JSON.stringify({
      path,
      maps,
      failures: triangulationFailures(annotation),
      failureRates: failureRates(annotation),
    }),
  );
}
