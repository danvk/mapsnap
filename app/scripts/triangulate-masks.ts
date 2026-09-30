// Print one JSON line per annotation file: {path, maps, failures}.
//   node scripts/triangulate-masks.ts <annotation.json>...
import fs from 'node:fs';
import { parseAnnotation } from '@allmaps/annotation';
import { triangulationFailures } from '../server/maskTriangulation.ts';

for (const path of process.argv.slice(2)) {
  const annotation = JSON.parse(fs.readFileSync(path, 'utf8'));
  const maps = parseAnnotation(annotation).length;
  const failures = triangulationFailures(annotation);
  console.log(JSON.stringify({ path, maps, failures }));
}
